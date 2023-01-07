import os
import json
import torch
import random
from torch import nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, get_scheduler

num_epoch = 30
batch_size = 4

scaler = GradScaler()
is_autocast = True

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

def split_data(data, eval_ratio=0.01):
    random.seed(0)
    random.shuffle(data)
    split_idx = int(len(data)*eval_ratio)
    eval_set = data[:split_idx]
    train_set = data[split_idx:]
    return train_set, eval_set

def collate_batch(batch):
  ids_list, attention_list = [], []
  for (_ids,_attention) in batch:
    ids_list.append(_ids)
    attention_list.append(_attention)
  input_ids_list = torch.nn.utils.rnn.pad_sequence(ids_list, batch_first=True)
  attention_list = torch.nn.utils.rnn.pad_sequence(attention_list, batch_first=True)
  output_ids_list = torch.nn.utils.rnn.pad_sequence(ids_list, batch_first=True, padding_value=-100)

  return input_ids_list.to(device), attention_list.to(device), output_ids_list.to(device)

class MyTokenizer():
    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.merged_tokens = []
        for i in range(8):
            self.merged_tokens.append('[SECS_'+str(i+1)+']')
        for i in range(32):
            self.merged_tokens.append('[BARS_'+str(i+1)+']')
        for i in range(11):
            self.merged_tokens.append('[SIM_'+str(i)+']')

    def __len__(self):
        return 128+len(self.merged_tokens)
    
    def encode(self, text):
        encodings = {}
        encodings['input_ids'] = torch.tensor(self.txt2ids(text, self.merged_tokens))
        encodings['attention_mask'] = torch.tensor([1]*len(encodings['input_ids']))
        return encodings

    def decode(self, ids):
        txt = ""
        for i in ids:
            if i>=128:
                txt += self.merged_tokens[i-128]
            else:
                txt += chr(i)
        return txt

    def txt2ids(self, text, merged_tokens):
        ids = [str(ord(c)) for c in text]
        txt_ids = ' '.join(ids)
        for t_idx, token in enumerate(merged_tokens):
            token_ids = [str(ord(c)) for c in token]
            token_txt_ids = ' '.join(token_ids)
            txt_ids = txt_ids.replace(token_txt_ids, str(t_idx+128))
        
        txt_ids = txt_ids.split(' ')
        txt_ids = [int(i) for i in txt_ids]
        return [self.bos_token_id]+txt_ids+[self.eos_token_id]
    
class MyDataset(Dataset):
    def __init__(self, items, tokenizer, max_length=1024):
        self.input_ids = []
        self.input_masks = []

        for item in tqdm(items):
            cc_tune = item["cc"]+item["tune"]
            tune_encodings = tokenizer.encode(cc_tune)
            if len(tune_encodings['input_ids']) <= max_length:
                self.input_ids.append(tune_encodings['input_ids'])
                self.input_masks.append(tune_encodings['attention_mask'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.input_masks[idx]

with open('abc_notation_cc.json') as f:
    data = json.load(f)
    train_set, eval_set = split_data(data)
    data = []

tokenizer = MyTokenizer()
train_set = DataLoader(MyDataset(train_set, tokenizer), batch_size=batch_size, collate_fn=collate_batch)
eval_set = DataLoader(MyDataset(eval_set, tokenizer), batch_size=batch_size, collate_fn=collate_batch)
config = GPT2Config(vocab_size=len(tokenizer))
model = GPT2LMHeadModel(config).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=1000,
    num_training_steps=num_epoch * len(train_set),
)

# call model with a batch of input
def process_one_batch(batch):
    b_input_ids = batch[0].to(device)
    b_masks = batch[1].to(device)
    b_labels = batch[2].to(device)
    
    try:
        loss = model(input_ids=b_input_ids,
                    attention_mask=b_masks,
                    labels=b_labels).loss
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print("WARNING: out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            return None
        else:
            raise exception
    return loss.mean()

# do one epoch for training
def train_epoch():
    tqdm_train_set = tqdm(train_set)
    total_train_loss = 0
    iter_idx = 1
    model.train()

    for batch in tqdm_train_set:
        if is_autocast:
            with autocast():
                loss = process_one_batch(batch)
            if loss==None:
                continue
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = process_one_batch(batch)
            if loss==None:
                continue
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        model.zero_grad(set_to_none=True)
        total_train_loss += loss.item()
        tqdm_train_set.set_postfix({'train_loss': total_train_loss / iter_idx})
        iter_idx += 1
        
    return total_train_loss / (iter_idx-1)

# do one epoch for eval
def eval_epoch():
    tqdm_eval_set = tqdm(eval_set)
    total_eval_loss = 0
    iter_idx = 1
    model.eval()
  
    # Evaluate data for one epoch
    for batch in tqdm_eval_set: 
        with torch.no_grad():
            loss = process_one_batch(batch)
            total_eval_loss += loss.item()
        tqdm_eval_set.set_postfix({'eval_loss': total_eval_loss / iter_idx})
        iter_idx += 1

    return total_eval_loss / (iter_idx-1)

# train and eval
if __name__ == "__main__":
    best_epoch = 0
    min_eval_loss = 100

    for i in range(num_epoch):
        print('-' * 21 + "Epoch " + str(i+1) + '-' * 21)
        train_loss = train_epoch()
        eval_loss = eval_epoch()
        with open('log.txt','a') as f:
            f.write("Epoch " + str(i+1) + "\ntrain_loss: " + str(train_loss) + "\neval_loss: " +str(eval_loss) + "\n\n")  
        if eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            best_epoch = i
            if torch.cuda.device_count() > 1:
                model.module.save_pretrained('weights')
            else:
                model.save_pretrained('weights')

    print("best epoch : "+str(best_epoch+1))
    print("min_eval_loss : "+str(min_eval_loss))