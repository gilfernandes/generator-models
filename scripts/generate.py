import torch
import numpy as np
from model import GPTLanguageModel
from config import Config
from simple_tokenizer import read_chars

import click

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def load_model(cfg: Config, path: str) -> GPTLanguageModel:
    model = GPTLanguageModel(cfg)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model.to(device)


def generate_decode(chars):
    itoi = {i: ch for i, ch in enumerate(chars)}
    return lambda  l: ''.join([itoi[i] for i in l])


def generate_encode(chars):
    stoi = {ch: i for i, ch in enumerate(chars)}
    return lambda s: [stoi[c] for c in s]


def generate(max_new_tokens: int, config_name: str, model_name: str, initial_input: str) -> str:
    cfg = Config(config_name)
    input_file = cfg.data_folder/cfg.input_file
    text, chars, vocab_size = read_chars(input_file)

    decode_func = generate_decode(chars)

    cfg.vocab_size = vocab_size
    model = load_model(cfg, f'../models/{model_name}')
    if initial_input is None:
        idx = torch.zeros([1, 1]).long()
    else:
        encode_func = generate_encode(chars)
        idx = torch.tensor(encode_func(initial_input)).unsqueeze(1)
    idx = idx.to(device)
    return decode_func(model.generate(idx, max_new_tokens)[0].tolist())
    
@click.command()
@click.option('--max_new_tokens', prompt="Please enter the maximum amount of tokens", default=300, help='Number of tokens to generate')
@click.option('--initial_tokens', prompt="Please enter a sentence to start with", default='.', help='Initial sentence to start with')
@click.option('--config_name', prompt="Please enter the configuration to use", default='config_522_512.yaml', help='The configuration file to use')
@click.option('--model', 
              prompt="Please enter the model file to use", 
              default='gpt_block_512_nlayer_8_nhead_6_iters_4000_03_11_2023_15_22_24_config_522_512.yaml_tensor(0.9748)_tensor(1.4976).pth', 
              help='The configuration file to use'
             )
def process_generation(max_new_tokens: int, config_name: str, model: str, initial_tokens: str) -> str:
    return print(generate(max_new_tokens, config_name, model, initial_tokens))

"""
Examples:

python generate.py --max_new_tokens 1000 --initial_tokens Love --config_name config_522_512.yaml --model gpt_block_512_nlayer_8_nhead_6_iters_4000_03_11_2023_15_22_24_config_522_512.yaml_tensor\(0.9748\)_tensor\(1.4976\).pth

python generate.py --max_new_tokens 1000 --initial_tokens Liebe --config_name config_522_512_goethe.yaml --model gpt_block_512_nlayer_8_nhead_6_iters_8000_03_29_2023_10_12_50_config_522_512_goethe.yaml_tensor\(1.0212\)_tensor\(1.2319\).pth

python generate.py --max_new_tokens 1000 --initial_tokens Liebe --config_name config_522_768.yaml --model gpt_block_768_nlayer_8_nhead_6_iters_4000_04_04_2023_10_58_27_config_522_768.yaml_tensor\(1.1021\)_tensor\(1.4717\).pth

"""
if __name__ == "__main__":
    process_generation()