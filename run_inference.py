import torch
from samplings import top_p_sampling
from transformers import GPT2LMHeadModel

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

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

    def decode(self, ids, skip_special_tokens=False):
        txt = ""
        for i in ids:
            if i>=128:
                if not skip_special_tokens:
                    txt += self.merged_tokens[i-128]
            elif i!=2 and i!=3:
                txt += chr(i)
        return txt

    def txt2ids(self, text, merged_tokens):
        ids = [str(ord(c)) for c in text]
        if torch.max(torch.tensor([ord(c) for c in text]))>=128:
            return [128+len(self.merged_tokens)]
        txt_ids = ' '.join(ids)
        for t_idx, token in enumerate(merged_tokens):
            token_ids = [str(ord(c)) for c in token]
            token_txt_ids = ' '.join(token_ids)
            txt_ids = txt_ids.replace(token_txt_ids, str(t_idx+128))
        
        txt_ids = txt_ids.split(' ')
        txt_ids = [int(i) for i in txt_ids]
        return [self.bos_token_id]+txt_ids+[self.eos_token_id]

tokenizer = MyTokenizer()
model = GPT2LMHeadModel.from_pretrained('weights').to(device)

# generate a sentence
def generate_txt(prompt, num_return_sequences=10, max_length=1024, top_p=0.9):
    model.eval()
    if prompt:
        ids = tokenizer.encode(prompt)['input_ids'][:-1]
    else:
        ids = torch.tensor([tokenizer.bos_token_id])

    for c_idx in range(num_return_sequences):
        print("\nX:"+str(c_idx+1)+"\n", end="")
        print(tokenizer.decode(ids[1:], skip_special_tokens=True), end="")
        input_ids = ids.unsqueeze(0)
        for t_idx in range(max_length):
            outputs = model(input_ids=input_ids.to(device))
            probs = outputs.logits[0][-1]
            probs = torch.nn.Softmax(dim=-1)(probs).cpu().detach().numpy()
            sampled_id = top_p_sampling(probs, top_p=top_p)
            input_ids = torch.cat((input_ids, torch.tensor([[sampled_id]])), 1)
            if sampled_id!=tokenizer.eos_token_id:
                print(tokenizer.decode([sampled_id], skip_special_tokens=True), end="")
                continue
            else:
                tune = "X:"+str(c_idx+1)+"\n"+tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
                with open('output_tunes/'+str(c_idx+1)+".abc", "w") as f:
                    f.write(tune)
                print("\n")
                break

if __name__ == "__main__":
    control_codes = "[SECS_3][BARS_8][SIM_3][BARS_8][SIM_10][SIM_3][BARS_8]"
    prompt = """L:1/4
M:4/4
K:C
 "C" E3/2 D/"G" G3/2"C" E/ | c G E G |"G" D3/2 E/ F A |"G" A G"C" C2 | E3/2 D/"G" G3/2"C" E/ |
 c G E G |"G" D3/2 E/"D" F D |"G" A G"C" c2 ||"""
    generate_txt(control_codes+prompt)