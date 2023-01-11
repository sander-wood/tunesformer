import os
import time
import torch
import random
import argparse
from transformers import GPT2LMHeadModel
from samplings import top_p_sampling, temperature_sampling

class ABCTokenizer():
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

def generate_abc(prompt, args):
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0), '\n')
    else:
        print('No GPU available, using the CPU instead.\n')
        device = torch.device("cpu")

    num_tunes = args.num_tunes
    max_length = args.max_length
    top_p = args.top_p
    temperature = args.temperature
    seed = args.seed
    print(" HYPERPARAMETERS ".center(60, "#"), '\n')
    args = vars(args)
    for key in args.keys():
        print(key+': '+str(args[key]))

    tokenizer = ABCTokenizer()
    # model = GPT2LMHeadModel.from_pretrained('weights').to(device)
    model = GPT2LMHeadModel.from_pretrained('sander-wood/tunesformer').to(device)

    if prompt:
        ids = tokenizer.encode(prompt)['input_ids'][:-1]
    else:
        ids = torch.tensor([tokenizer.bos_token_id])

    random.seed(seed)
    tunes = ""
    print("\n"+" OUTPUT TUNES ".center(60, "#"))

    for c_idx in range(num_tunes):
        print("\nX:"+str(c_idx+1)+"\n", end="")
        print(tokenizer.decode(ids[1:], skip_special_tokens=True), end="")
        input_ids = ids.unsqueeze(0)
        for t_idx in range(max_length):
            if seed!=None:
                n_seed = random.randint(0, 1000000)
                random.seed(n_seed)
            else:
                n_seed = None

            outputs = model(input_ids=input_ids.to(device))
            probs = outputs.logits[0][-1]
            probs = torch.nn.Softmax(dim=-1)(probs).cpu().detach().numpy()
            sampled_id = temperature_sampling(probs=top_p_sampling(probs, 
                                                                top_p=top_p, 
                                                                seed=n_seed,
                                                                return_probs=True),
                                            seed=n_seed,
                                            temperature=temperature)
            input_ids = torch.cat((input_ids, torch.tensor([[sampled_id]])), 1)
            if sampled_id!=tokenizer.eos_token_id:
                print(tokenizer.decode([sampled_id], skip_special_tokens=True), end="")
                continue
            else:
                tune = "X:"+str(c_idx+1)+"\n"+tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
                tunes += tune+"\n\n"
                print("\n")
                break

    timestamp = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime()) 
    with open('output_tunes/'+timestamp+'.abc', 'w') as f:
        f.write(tunes)

def get_args(parser):

    parser.add_argument('-num_tunes', type=int, default=3, help='the number of independently computed returned tunes')
    parser.add_argument('-max_length', type=int, default=1024, help='integer to define the maximum length in tokens of each tune')
    parser.add_argument('-top_p', type=float, default=0.9, help='float to define the tokens that are within the sample operation of text generation')
    parser.add_argument('-temperature', type=float, default=1., help='the temperature of the sampling operation')
    parser.add_argument('-seed', type=int, default=None, help='seed for randomstate')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    control_codes = "[SECS_3][BARS_4][SIM_6][BARS_4][SIM_10][SIM_6][BARS_4]"
    prompt = """L:1/4
M:4/4
K:C
"C" C C G G |"F" A A"C" G2 |"G" F F"C" E E |"G" D D"C" C2 ||"""
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    generate_abc(control_codes+prompt, args)