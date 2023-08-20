import time
import torch
from utils import *
import re
from config import *
from transformers import  GPT2Config
import argparse
import os
from tqdm import tqdm
import requests

def get_args(parser):

    parser.add_argument('-num_tunes', type=int, default=3, help='the number of independently computed returned tunes')
    parser.add_argument('-max_patch', type=int, default=128, help='integer to define the maximum length in tokens of each tune')
    parser.add_argument('-top_p', type=float, default=0.8, help='float to define the tokens that are within the sample operation of text generation')
    parser.add_argument('-top_k', type=int, default=8, help='integer to define the tokens that are within the sample operation of text generation')
    parser.add_argument('-temperature', type=float, default=1.2, help='the temperature of the sampling operation')
    parser.add_argument('-seed', type=int, default=None, help='seed for randomstate')
    parser.add_argument('-show_control_code', type=bool, default=True, help='whether to show control code')
    args = parser.parse_args()

    return args

def generate_abc(args):

    if torch.cuda.is_available():    
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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

    filename = "weights.pth"

    if os.path.exists(filename):
        print(f"Weights already exist at '{filename}'. Loading...")
    else:
        print(f"Downloading weights to '{filename}' from huggingface.co...")
        try:
            url = 'https://huggingface.co/sander-wood/tunesformer/resolve/main/weights.pth'
            response = requests.get(url, stream=True)

            # 获取文件大小
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 1024

            # 使用tqdm来显示进度条
            with open(filename, 'wb') as file, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=chunk_size):
                    size = file.write(data)
                    bar.update(size)
        except Exception as e:
            print(f"Error: {e}")
            exit()
            
    checkpoint = torch.load('weights.pth')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    with open('prompt.txt', 'r') as f:
        prompt = f.read()

    tunes = ""
    num_tunes = args.num_tunes
    max_patch = args.max_patch
    top_p = args.top_p
    top_k = args.top_k
    temperature = args.temperature
    seed = args.seed
    show_control_code = args.show_control_code

    print(" HYPERPARAMETERS ".center(60, "#"), '\n')
    args = vars(args)
    for key in args.keys():
        print(key+': '+str(args[key]))
    print("\n"+" OUTPUT TUNES ".center(60, "#"))

    start_time = time.time()

    for i in range(num_tunes):
        tune = "X:"+str(i+1) + "\n" + prompt
        lines = re.split(r'(\n)', tune)
        tune = ""
        skip = False
        for line in lines:
            if show_control_code or line[:2] not in ["S:", "B:", "E:"]:
                if not skip:
                    print(line, end="")
                    tune += line
                skip = False
            else:
                skip = True

        input_patches = torch.tensor([patchilizer.encode(prompt, add_special_patches=True)[:-1]], device=device)
        if tune=="":
            tokens = None
        else:
            prefix = patchilizer.decode(input_patches[0])
            remaining_tokens = prompt[len(prefix):]
            tokens  = torch.tensor([patchilizer.bos_token_id]+[ord(c) for c in remaining_tokens], device=device)
        
        while input_patches.shape[1]<max_patch:
            predicted_patch, seed = model.generate(input_patches,
                                                    tokens,
                                                    top_p=top_p,
                                                    top_k=top_k,
                                                    temperature=temperature,
                                                    seed=seed)
            tokens = None
            if predicted_patch[0]!=patchilizer.eos_token_id:
                next_bar = patchilizer.decode([predicted_patch])
                if show_control_code or next_bar[:2] not in ["S:", "B:", "E:"]:
                    print(next_bar, end="")
                    tune += next_bar
                if next_bar=="":
                    break
                next_bar = remaining_tokens+next_bar
                remaining_tokens = ""
                predicted_patch = torch.tensor(patchilizer.bar2patch(next_bar), device=device).unsqueeze(0)
                input_patches = torch.cat([input_patches, predicted_patch.unsqueeze(0)], dim=1)
            else:
                break

        tunes += tune+"\n\n"
        print("\n")

    print("Generation time: {:.2f} seconds".format(time.time()-start_time))
    timestamp = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime()) 
    with open('output_tunes/'+timestamp+'.abc', 'w') as f:
        f.write(tunes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = get_args(parser)
    generate_abc(args)