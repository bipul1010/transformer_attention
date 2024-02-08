from typing import ValuesView
import torch
from config import get_config


class KVCache:
    def __init__(self, max_batch_size, max_seq_len, head_dim, device) -> None:
        self.cache_k = torch.zeros(max_batch_size, max_seq_len, head_dim).to(device)
        self.cache_v = torch.zeros(max_batch_size, max_seq_len, head_dim).to(device)

    def update(self, xk, xv, batch_size, start_pos):
        self.cache_k[:batch_size, start_pos : start_pos + xk.shape[1], :] = xk
        self.cache_v[:batch_size, start_pos : start_pos + xv.shape[1], :] = xv

    def get(self, batch_size, start_pos):
        keys = self.cache_k[:batch_size, :start_pos]
        values = self.cache_v[:batch_size, :start_pos]
        return keys, values
