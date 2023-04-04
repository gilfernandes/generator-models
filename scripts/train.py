import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
from model import GPTLanguageModel
import sys
import re

import pickle
from config import Config

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

from datetime import datetime
from simple_tokenizer import read_chars, create_vocab_dicts



def train_test_split(text, split_factor=0.9):
    # Train / test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(split_factor * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data



# data loading
def get_batch(split):
    block_size = cfg.block_size
    batch_size = cfg.batch_size
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(cfg.device), y.to(cfg.device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def generate_identifier(cfg: Config):
    now = datetime.now()
    timestamp = now.strftime("%m_%d_%Y_%H_%M_%S")
    return f"block_{cfg.block_size}_nlayer_{cfg.n_layer}_nhead_{cfg.n_head}_iters_{cfg.max_iters}_{timestamp}_{cfg.name}"


def save_model(model, cfg, losses):
    logging.info("Saving model")
    identifier = generate_identifier(cfg)
    if losses is not None:
        identifier += f"_{str(losses['train'])}_{str(losses['val'])}"
    model_file = f"../models/gpt_{identifier}.pth"
    torch.save(model.state_dict(), model_file)
    logging.info(f"Model saved to {model_file}")
    
    
    
def generate_sample(model, cfg):
    logging.info("Starting generation")
    context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
    generated_text = decode(model.generate(context, max_new_tokens=cfg.generated_characters)[0].tolist())
    logging.info(generated_text)
    
    identifier = generate_identifier(cfg)

    with open(f'../generations/output_{identifier}.txt', 'w') as f:
        f.write(generated_text)


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        logging.error("Please enter the configuration file.")
        quit()
    
    cfg = Config(sys.argv[1])
    torch.manual_seed(cfg.manual_seed)

    text, chars, vocab_size = read_chars(cfg.data_folder/cfg.input_file)
    cfg.vocab_size = vocab_size

    # create a mapping of characters to integers
    stoi, itoi = create_vocab_dicts(chars)
    def encode(s): return [stoi[c] for c in s]
    def decode(l): return ''.join([itoi[i] for i in l])

    # Write the dictionary with the key char codes
    def remove_extension(file_name):
        return re.sub(r"(.+)\..+", r"\1", file_name)
        
    with open(f'{remove_extension(cfg.input_file)}_itoi.pickle', 'wb') as handle:
        pickle.dump(itoi, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    train_data, val_data = train_test_split(text)
    
    model = GPTLanguageModel(cfg)
    model = model.to(cfg.device)
    
    # create a Pytorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    logging.info("Training Started")
    
    for iter in range(cfg.max_iters):
        # every once and then evaluate the loss on train and val sets
        if iter % cfg.eval_interval == 0:
            losses = estimate_loss(model)
            logging.info(
                f"step ({iter}): train loss: {losses['train']}, val loss: {losses['val']}")
            if losses['val'] < cfg.model_save_threshold:
                save_model(model, cfg, losses)

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    logging.info("Finished training")

    logging.info("Saving model")
    save_model(model, cfg, estimate_loss(model))
    
    generate_sample(model, cfg)


