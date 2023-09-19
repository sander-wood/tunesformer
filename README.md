# TunesFormer

## Model description

TunesFormer is an efficient Transformer-based dual-decoder model specifically designed for the generation of melodies that adhere to user-defined musical forms. It was introduced in the paper [TunesFormer: Forming Irish Tunes with Control Codes by Bar Patching](https://arxiv.org/abs/2301.02884) by Wu et al. The [model](https://huggingface.co/sander-wood/tunesformer) and the [dataset](https://huggingface.co/datasets/sander-wood/irishman) is released in huggingface. 

Trained on 214,122 Irish tunes, TunesFormer utilizes techniques including bar patching and control codes. Bar patching reduces sequence length and generation time, while control codes guide TunesFormer in producing melodies that conform to desired musical forms.

TunesFormer (GPT-2 version) is available for online use and experience on [huggingface spaces](https://huggingface.co/spaces/sander-wood/tunesformer). For the full dual-decoder version of TunesFormer, please use the scripts from the [official GitHub repository](https://github.com/sander-wood/tunesformer).


## ABC Notation

ABC notation is an ASCII-based plain text musical notation system that is commonly used for transcribing traditional music and sharing sheet music online. It provides a simple and concise way to represent musical elements such as notes, rhythms, chords, and more.

For those looking to interact with ABC notation in various ways, there are several tools available:

1. **[Online ABC Player](https://abc.rectanglered.com/):** This web-based tool allows you to input ABC notation and hear the corresponding audio playback. By pasting your ABC code into the player, you can instantly listen to the tune as it would sound when played.

2. **[ABC Sheet Music Editor - EasyABC](https://easyabc.sourceforge.net/):** EasyABC is a user-friendly software application designed for creating, editing, and formatting ABC notation. Its graphical interface enables you to input your ABC code, preview the sheet music, and make adjustments as necessary.

## Dataset

The **Irish Massive ABC Notation ([IrishMAN](https://huggingface.co/datasets/sander-wood/irishman))** dataset includes 216,284 Irish tunes in ABC notation, divided into 99\% (214,122 tunes) for training and 1\% (2,162 tunes) for validation. These tunes were collected from thesession.org and abcnotation.com, both renowned for sharing traditional music. To ensure uniformity in formatting, all tunes were converted to XML and then back to ABC using [scripts](https://wim.vree.org/svgParse/), and fields containing natural language (e.g., titles and lyrics) were removed.

Each tune is automatically annotated with control codes derived from ABC symbols, as described in the below section. These control codes offer insights into the musical forms and structures of each composition.

In the IrishMAN dataset, a [music21](https://web.mit.edu/music21/doc/index.html#)-filtered [subset](https://huggingface.co/datasets/sander-wood/irishman/raw/main/leadsheet_ids.json) includes 34,211 lead sheets, each human-annotated with chord symbols. It is from this very subset that TunesFormer developed its capacity to generate melodies with harmonies.

A noteworthy aspect is the copyright status. All tunes in the dataset are in the public domain, ensuring ethical and legal usage for research and creative projects.

## Control codes

Inspired by [CTRL](https://huggingface.co/ctrl), we incorporate control codes into TunesFormer to represent musical forms. These codes, positioned ahead of the ABC notation, enable users to specify the structures of the generated tunes. The following control codes are introduced:

- **S:number of sections**: determines the number of sections in the entire melody. It counts on several symbols that can be used to represent section boundaries: `[|`, `||`, `|]`, `|:`, `::`, and `:|`. In our dataset, the range is 1 to 8 (e.g., `S:1` for a single-section melody, and `S:8` for a melody with eight sections).

- **B:number of bars**: specifies the desired number of bars within a section. It counts on the bar symbol `|`. In our dataset, the range is 1 to 32 (e.g., `B:1` for a one-bar section, and `B:32` for a section with 32 bars).

- **E:edit distance similarity**: controls the similarity level between the current section $c$ and a previous section $p$ in the melody. It is based on the Levenshtein distance $lev(c,p)$ , quantifying the difference between sections for creating variations or contrasts. Mathematically, it can be expressed as:
  ```
  eds(c,p) = 1 - lev(c,p) / max(|c|,|p|)
  ```
  where $|c|$ and $|p|$ are the string lengths of the two sections. It is discretized into 11 levels, ranging from no match at all to an exact match (e.g., `E:0` for no similarity, and `E:10` for an exact match).

### How to use

1. Install dependencies for the code released in [this repository](https://github.com/sander-wood/tunesformer):
```
unidecode                    1.3.6
torch                        1.13.1+cu116
samplings                    0.1.7
transformers                 4.18.0
```

2. Set the prompt in `prompt.txt` for conditional music generation. 
```
S:2
B:9
E:4
B:9
L:1/8
M:3/4
K:D
 de |"D"
```
 
3. Run the script `generate.py`. When running a script for the first time, the downloaded weights will be cached for future reuse. If the automatic download does not work, you can manually download the weights from [here](https://huggingface.co/sander-wood/tunesformer/blob/main/weights.pth).

```
python generate.py -num_tunes 3 -max_patch 128 -top_p 0.8 -top_k 8 -temperature 1.2 -seed 0 -show_control_code True
```

4. Enjoy tunes in the folder `output_tunes`! If you want to convert these ABC tunes to sheet music or audio, please refer to `ABC Notation`.
```
X:1
S:2
B:9
E:4
B:9
L:1/8
M:3/4
K:D
 de |"D" f2 fedB | A2 AF A2 |"G" B2 Bd B2 |"D" A2 AF A2 |"D" f2 fedB | A2 AF A2 |"G" B2 Bd"A" ce |
"D" d4 :: de |"D" f2 fdfa |"Em" g2 gfed |"A7" cecA ce |"D" fdfafd |"D" f2 fdfa |"Em" g2 gfed |
"A7" cecA ce |"D" d4 :|

X:2
S:2
B:9
E:4
B:9
L:1/8
M:3/4
K:D
 de |"D" fdcdBd |"D" AdFAdf |"A" gecAce |"D" fdAdde | fdcdBd |"D" AdFAdf |"A" gecAce |"D" d4 :: a2 |
"G" gfgbd'b |"D" fad'fad' |"A7" c'bagfe |"D" d'afd A2 |"G" gfgbd'b |"D" fad'fad' |"A7" c'bagfe |
"D" d4 :|

X:3
S:2
B:9
E:4
B:9
L:1/8
M:3/4
K:D
 de |"D" f3 e dc |"G" d2 cB AG |"D" F2 A2 d2 |"A7" f4 e2 |"D" f3 e dc |"G" d2 c2 B2 |"A7" A2 B2 c2 |
"D" d4 :| FG |"D" A3 B A2 |"D" A3 G FG |"D" A2 d2 f2 |"A7" e4 AA |"G" B2 Bc de |"D" f2 d2 A2 |
"G" Bc d2"A7" c2 |"D" d4 :|
```

### Caution
ABC notation is a specialized notation of representing sheet music, and it follows a specific standard format. When interacting with TunesFormer, all trained ABC notation adheres to these standard formats.

If you are unfamiliar with the details of ABC notation, we strongly recommend against manually entering ABC notation. Otherwise, the model may not recognize and generate the music correctly. Inputting incorrect formats might lead to unpredictable outputs or other issues.

A general recommendation is to adjust the desired musical structure and form through control codes and ABC header, rather than directly editing the ABC notation itself.

For more detailed information about the ABC notation standard, you can refer to the [official ABC notation standard description](https://abcnotation.com/wiki/abc:standard:v2.1).

Please make sure to operate according to the provided formats and guidelines to fully leverage the capabilities of TunesFormer and achieve a satisfying music generation experience.

### Use your own dataset
Follow these steps:

1. First, unzip the `data_curation.zip` file.
2. Next, place your own symbol music data (in .xml, .mxl, .musicxml format) into the `xmls` folder.
3. Run the `batch_converter.py` file to convert your data to ABC notation into the `abcs` folder.
4. Run the `add_control_codes.py` file. This will add control codes to your ABC notation and save the data as `dataset.json`.
5. Finally, run the `train.py` file to train your own model. The weights will be saved as `weights.pth`

### Usage
```
usage: generate.py [-h] [-num_tunes NUM_TUNES] [-max_patch MAX_PATCH]
                   [-top_p TOP_P] [-top_k TOP_K] [-temperature TEMPERATURE]
                   [-seed SEED] [-show_control_code SHOW_CONTROL_CODE]

optional arguments:
  -h, --help            show this help message and exit
  -num_tunes NUM_TUNES  the number of independently computed returned tunes
  -max_patch MAX_PATCH  integer to define the maximum length in tokens of each
                        tune
  -top_p TOP_P          float to define the tokens that are within the sample
                        operation of text generation
  -top_k TOP_K          integer to define the tokens that are within the
                        sample operation of text generation
  -temperature TEMPERATURE
                        the temperature of the sampling operation
  -seed SEED            seed for randomstate
  -show_control_code SHOW_CONTROL_CODE
                        whether to show control code
```

### BibTeX entry and citation info

```bibtex
@misc{https://doi.org/10.48550/arxiv.2301.02884,
  title = {TunesFormer: Forming Irish Tunes with Control Codes by Bar Patching},
      author={Shangda Wu and Xiaobing Li and Feng Yu and Maosong Sun},
      year={2023},
      eprint={2301.02884},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
