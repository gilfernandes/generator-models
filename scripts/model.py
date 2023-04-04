import torch
import torch.nn as nn
from config import Config

from torch.nn import functional as F

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, cfg: Config):
        super().__init__()
        self.key = nn.Linear(cfg.n_embed, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embed, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

        self.dropout = nn.Dropout(cfg.dropout)
        self.head_size = head_size

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        # wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        return wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, cfg: Config):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, cfg) for _ in range(num_heads)])
        self.proj = nn.Linear(cfg.n_embed, cfg.n_embed)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head, cfg: Config) -> None:
        # n_embed: embedding dimension, n_head: the number of heads we would like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size, cfg)
        self.ffwd = FeedForward(n_embed, cfg)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embed)
        self.blocks = nn.Sequential(
            *[Block(cfg.n_embed, cfg.n_head, cfg) for _ in range(cfg.n_layer)]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embed)
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size)
        self.cfg = cfg

    def forward(self, idx, targets=None):

        B, T = idx.shape
        # (B, T, C) (4, 8, cfg.n_embed) (batch, sequence length, embed size)
        token_embeddings = self.token_embedding_table(idx)  # (B, T, C)
        position_embeddings = self.position_embedding_table(
            torch.arange(T, device=self.cfg.device))  # (T, C)
        x = token_embeddings + position_embeddings  # (B, T, C)
        x = self.blocks(x)  # apply one head of self-attention (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, C (vocab_size))

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) (batch, sequence length) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.cfg.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)  # Loss is ignored
            # focus only on one timestep
            logits = logits[:, -1, :]  # (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx
