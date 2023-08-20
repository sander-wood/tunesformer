import re
import torch
import random
from config import *
from unidecode import unidecode
from samplings import top_p_sampling, top_k_sampling, temperature_sampling
from transformers import GPT2Model, GPT2LMHeadModel, PreTrainedModel

class Patchilizer:
    """
    A class for converting music bars to patches and vice versa. 
    """
    def __init__(self):
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = '(' + '|'.join(map(re.escape, self.delimiters)) + ')'
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

    def split_bars(self, body):
        """
        Split a body of music into individual bars.
        """
        bars = re.split(self.regexPattern, ''.join(body))
        bars = list(filter(None, bars))  # remove empty strings
        if bars[0] in self.delimiters:
            bars[1] = bars[0] + bars[1]
            bars = bars[1:]
        bars = [bars[i * 2] + bars[i * 2 + 1] for i in range(len(bars) // 2)]
        return bars
    
    def bar2patch(self, bar, patch_size=PATCH_SIZE):
        """
        Convert a bar into a patch of specified length.
        """
        patch = [self.bos_token_id] + [ord(c) for c in bar] + [self.eos_token_id]
        patch = patch[:patch_size]
        patch += [self.pad_token_id] * (patch_size - len(patch))
        return patch
    
    def patch2bar(self, patch):
        """
        Convert a patch into a bar.
        """
        return ''.join(chr(idx) if idx > self.eos_token_id else '' for idx in patch if idx != self.eos_token_id)

    def encode(self, abc_code, patch_length=PATCH_LENGTH, patch_size=PATCH_SIZE, add_special_patches=False):
        """
        Encode music into patches of specified length.
        """
        lines = unidecode(abc_code).split('\n')
        lines = list(filter(None, lines))  # remove empty lines

        body = ""
        patches = []

        for line in lines:
            if len(line) > 1 and ((line[0].isalpha() and line[1] == ':') or line.startswith('%%score')):
                if body:
                    bars = self.split_bars(body)
                    patches.extend(self.bar2patch(bar + '\n' if idx == len(bars) - 1 else bar, patch_size) 
                                   for idx, bar in enumerate(bars))
                    body = ""
                patches.append(self.bar2patch(line + '\n', patch_size))
            else:
                body += line + '\n'

        if body:
            patches.extend(self.bar2patch(bar, patch_size) for bar in self.split_bars(body))

        if add_special_patches:
            bos_patch = [self.bos_token_id] * (patch_size-1) + [self.eos_token_id]
            eos_patch = [self.bos_token_id] + [self.eos_token_id] * (patch_size-1)
            patches = [bos_patch] + patches + [eos_patch]

        return patches[:patch_length]

    def decode(self, patches):
        """
        Decode patches into music.
        """
        return ''.join(self.patch2bar(patch) for patch in patches)

class PatchLevelDecoder(PreTrainedModel):
    """
    An Patch-level Decoder model for generating patch features in an auto-regressive manner. 
    It inherits PreTrainedModel from transformers.
    """

    def __init__(self, config):
        super().__init__(config)
        self.patch_embedding = torch.nn.Linear(PATCH_SIZE * 128, config.n_embd)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.base = GPT2Model(config)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the patch-level decoder model.
        :param patches: the patches to be encoded
        :return: the encoded patches
        """
        patches = torch.nn.functional.one_hot(patches, num_classes=128).float()
        patches = patches.reshape(len(patches), -1, PATCH_SIZE * 128)
        patches = self.patch_embedding(patches.to(self.device))

        return self.base(inputs_embeds=patches)

class CharLevelDecoder(PreTrainedModel):
    """
    A Char-level Decoder model for generating the characters within each bar patch sequentially. 
    It inherits PreTrainedModel from transformers.
    """
    def __init__(self, config):
        super().__init__(config)
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.base = GPT2LMHeadModel(config)

    def forward(self, encoded_patches: torch.Tensor, target_patches: torch.Tensor, patch_sampling_batch_size: int):
        """
        The forward pass of the char-level decoder model.
        :param encoded_patches: the encoded patches
        :param target_patches: the target patches
        :return: the decoded patches
        """
        # preparing the labels for model training
        target_masks = target_patches == self.pad_token_id
        labels = target_patches.clone().masked_fill_(target_masks, -100)

        # masking the labels for model training
        target_masks = torch.ones_like(labels)
        target_masks = target_masks.masked_fill_(labels == -100, 0)

        # select patches
        if patch_sampling_batch_size!=0 and patch_sampling_batch_size<target_patches.shape[0]:
            indices = list(range(len(target_patches)))
            random.shuffle(indices)
            selected_indices = sorted(indices[:patch_sampling_batch_size])

            target_patches = target_patches[selected_indices,:]
            target_masks = target_masks[selected_indices,:]
            encoded_patches = encoded_patches[selected_indices,:]
            labels = labels[selected_indices,:]

        # get input embeddings
        inputs_embeds = torch.nn.functional.embedding(target_patches, self.base.transformer.wte.weight)

        # concatenate the encoded patches with the input embeddings
        inputs_embeds = torch.cat((encoded_patches.unsqueeze(1), inputs_embeds[:,1:,:]), dim=1)

        return self.base(inputs_embeds=inputs_embeds,
                         attention_mask=target_masks,
                         labels=labels)

    def generate(self, encoded_patch: torch.Tensor, tokens: torch.Tensor):
        """
        The generate function for generating a patch based on the encoded patch and already generated tokens.
        :param encoded_patch: the encoded patch
        :param tokens: already generated tokens in the patch
        :return: the probability distribution of next token
        """
        encoded_patch = encoded_patch.reshape(1, 1, -1)
        tokens = tokens.reshape(1, -1)

        # Get input embeddings
        tokens = torch.nn.functional.embedding(tokens, self.base.transformer.wte.weight)

        # Concatenate the encoded patch with the input embeddings
        tokens = torch.cat((encoded_patch, tokens[:,1:,:]), dim=1)
        
        # Get output from model
        outputs = self.base(inputs_embeds=tokens)
        
        # Get probabilities of next token
        probs = torch.nn.functional.softmax(outputs.logits.squeeze(0)[-1], dim=-1)

        return probs

class TunesFormer(PreTrainedModel):
    """
    TunesFormer is a hierarchical music generation model based on bar patching. 
    It includes a patch-level decoder and a character-level decoder.
    It inherits PreTrainedModel from transformers.
    """

    def __init__(self, encoder_config, decoder_config, share_weights=False):
        super().__init__(encoder_config)
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        if share_weights:
            max_layers = max(encoder_config.num_hidden_layers, decoder_config.num_hidden_layers)
            max_context_size = max(encoder_config.max_length, decoder_config.max_length)
            max_position_embeddings = max(encoder_config.max_position_embeddings, decoder_config.max_position_embeddings)

            encoder_config.num_hidden_layers = max_layers
            encoder_config.max_length = max_context_size
            encoder_config.max_position_embeddings = max_position_embeddings
            decoder_config.num_hidden_layers = max_layers
            decoder_config.max_length = max_context_size
            decoder_config.max_position_embeddings = max_position_embeddings

        self.patch_level_decoder = PatchLevelDecoder(encoder_config)
        self.char_level_decoder = CharLevelDecoder(decoder_config)

        if share_weights:
            self.patch_level_decoder.base = self.char_level_decoder.base.transformer

    def forward(self, patches: torch.Tensor, patch_sampling_batch_size: int=PATCH_SAMPLING_BATCH_SIZE):
        """
        The forward pass of the TunesFormer model.
        :param patches: the patches to be both encoded and decoded
        :return: the decoded patches
        """
        patches = patches.reshape(len(patches), -1, PATCH_SIZE)
        encoded_patches = self.patch_level_decoder(patches)["last_hidden_state"]
        
        return self.char_level_decoder(encoded_patches.squeeze(0)[:-1, :], patches.squeeze(0)[1:, :], patch_sampling_batch_size)

    def generate(self, patches: torch.Tensor,
                 tokens: torch.Tensor,
                 top_p: float=1,
                 top_k: int=0,
                 temperature: float=1,
                 seed: int=None):
        """
        The generate function for generating patches based on patches.
        :param patches: the patches to be encoded
        :return: the generated patches
        """
        patches = patches.reshape(len(patches), -1, PATCH_SIZE)
        encoded_patches = self.patch_level_decoder(patches)["last_hidden_state"]
        if tokens==None:
            tokens = torch.tensor([self.bos_token_id], device=self.device)
        generated_patch = []
        random.seed(seed)

        while True:
            if seed!=None:
                n_seed = random.randint(0, 1000000)
                random.seed(n_seed)
            else:
                n_seed = None
            prob = self.char_level_decoder.generate(encoded_patches[0][-1], tokens).cpu().detach().numpy()
            prob = top_p_sampling(prob, top_p=top_p, return_probs=True)
            prob = top_k_sampling(prob, top_k=top_k, return_probs=True)
            token = temperature_sampling(prob, temperature=temperature, seed=n_seed)
            generated_patch.append(token)
            if token == self.eos_token_id or len(tokens) >= PATCH_SIZE - 1:
                break
            else:
                tokens = torch.cat((tokens, torch.tensor([token], device=self.device)), dim=0)
        
        return generated_patch, n_seed