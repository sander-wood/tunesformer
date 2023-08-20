import os
import time
import json
import torch
import random
from utils import *
from config import *
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, get_scheduler

if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
random.seed(42)
batch_size = torch.cuda.device_count()
patchilizer = Patchilizer()

patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS, 
                    max_length=PATCH_LENGTH, 
                    max_position_embeddings=PATCH_LENGTH,
                    vocab_size=1)
char_config = GPT2Config(num_hidden_layers=CHAR_NUM_LAYERS, 
                    max_length=PATCH_SIZE, 
                    max_position_embeddings=PATCH_SIZE,
                    vocab_size=128)
model = TunesFormer(patch_config, char_config, share_weights=SHARE_WEIGHTS)

# print parameter number
print("Parameter Number: "+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

scaler = GradScaler()
is_autocast = True
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
def collate_batch(batch):
    input_patches = []
    
    for input_patch in batch:
        input_patches.append(input_patch.reshape(-1))

    input_patches = torch.nn.utils.rnn.pad_sequence(input_patches, batch_first=True, padding_value=0)

    return input_patches.to(device)

def split_data(data, eval_ratio=0.1):
    random.shuffle(data)
    split_idx = int(len(data)*eval_ratio)
    eval_set = data[:split_idx]
    train_set = data[split_idx:]
    return train_set, eval_set

class MyDataset(Dataset):
    def __init__(self, items):
        self.texts = []
        
        for item in tqdm(items):
            text = item['control code']+item['abc notation'][4:]
            input_patch =  patchilizer.encode(text, add_special_patches=True)
            input_patch = torch.tensor(input_patch)
            if torch.sum(input_patch)!=0:
                self.texts.append(input_patch)
            
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

# call model with a batch of input
def process_one_batch(batch):
    input_patches = batch
    
    loss = model(input_patches).loss

    return loss.mean()

# do one epoch for training
def train_epoch():
    tqdm_train_set = tqdm(train_set)
    total_train_loss = 0
    iter_idx = 1
    model.train()

    for batch in tqdm_train_set:
        try:
            if is_autocast:
                with autocast():
                    loss = process_one_batch(batch)
                if loss==None or torch.isnan(loss).item():
                    continue
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = process_one_batch(batch)
                if loss==None or torch.isnan(loss).item():
                    continue
                loss.backward()
                optimizer.step()
        except RuntimeError as exception:
            if "memory" in str(exception):
                print(str(exception))
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                continue
            else:
                raise exception
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
            if loss==None or torch.isnan(loss).item():
                continue
            total_eval_loss += loss.item()
        tqdm_eval_set.set_postfix({'eval_loss': total_eval_loss / iter_idx})
        iter_idx += 1

    return total_eval_loss / (iter_idx-1)

# train and eval
if __name__ == "__main__":
    
    # load data
    with open('data.json') as f:
        print("Loading Data...")
        data = json.load(f)
        train_set, eval_set = split_data(data)
        data = []

    train_set = DataLoader(MyDataset(train_set), batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
    eval_set = DataLoader(MyDataset(eval_set), batch_size=batch_size, collate_fn=collate_batch, shuffle=True)

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=NUM_EPOCHS * len(train_set) / 10,
        num_training_steps=NUM_EPOCHS * len(train_set),
    )
    if LOAD_FROM_CHECKPOINT and os.path.exists('weights.pth'):
        checkpoint = torch.load('weights.pth')
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_sched'])
        pre_epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        min_eval_loss = checkpoint['min_eval_loss']
        print("Successfully Loaded Checkpoint from Epoch %d" % pre_epoch)
    
    else:
        pre_epoch = 0
        best_epoch = 0
        min_eval_loss = 100

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, NUM_EPOCHS+1-pre_epoch):
        epoch += pre_epoch
        print('-' * 21 + "Epoch " + str(epoch) + '-' * 21)
        train_loss = train_epoch()
        eval_loss = eval_epoch()
        with open('logs.txt','a') as f:
            f.write("Epoch " + str(epoch) + "\ntrain_loss: " + str(train_loss) + "\neval_loss: " +str(eval_loss) + "\ntime: " + time.asctime(time.localtime(time.time())) + "\n\n")
        if eval_loss < min_eval_loss:
            best_epoch = epoch
            min_eval_loss = eval_loss
            if torch.cuda.device_count() > 1:
                checkpoint = { 
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_sched': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_epoch': best_epoch,
                'min_eval_loss': min_eval_loss}
            else:
                checkpoint = { 
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_sched': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_epoch': best_epoch,
                'min_eval_loss': min_eval_loss}
            torch.save(checkpoint, 'weights.pth')

    print("Best Eval Epoch : "+str(best_epoch))
    print("Min Eval Loss : "+str(min_eval_loss))