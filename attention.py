import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from dataset import generate_square_subsequent_mask


class TransformerAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.multi_head_attention = MultiheadAttention(
            embed_dim=d_model,
            num_heads=h,
            dropout=dropout,
            batch_first=True,
            bias=True,
        )

    def forward(self, query, key, value, key_padding_mask, attn_mask):
        attn_output, attn_output_weights = self.multi_head_attention(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )

        return attn_output


if __name__ == "__main__":
    d_model = 512
    h = 8
    dropout = 0.1
    batch_size = 4
    seq_length = 8

    mha = TransformerAttention(d_model=d_model, h=8, dropout=0.1)

    x = torch.rand(batch_size, seq_length, d_model)
    print(x)
    g = [False, False, False, False, False, True, True, True]
    key_padding_mask = torch.tensor([g for i in range(batch_size)])

    attn_mask = generate_square_subsequent_mask(seq_length)
    print("=========")
    output = mha(x, x, x, key_padding_mask, attn_mask=attn_mask)
    print(output)
    print(output.shape)
