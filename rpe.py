import torch.nn as nn
import torch


def pre_compute_complex_theta_pos(
    embed_dim: int, seq_len: int, theta_base: float = 10000
):
    theta_pow = torch.arange(0, embed_dim, 2)  ## embed_dim/2
    theta_vector = 1 / (theta_base ** (theta_pow / embed_dim))  ##embed_dim/2

    m = torch.arange(seq_len)  ##seq_len
    prod = torch.outer(m, theta_vector)  ## (seq_len,embed_dim/2)

    complex_theta_pos = torch.polar(torch.ones(*prod.shape), prod)

    ## (seq_len,embed_dim/2)
    return complex_theta_pos


class RotaryPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        ##(seq_len,d_model/2)
        com_theta_pos = pre_compute_complex_theta_pos(self.d_model, self.seq_len)   

        self.register_buffer("rpe", com_theta_pos)

    def forward(self, x):
        ## x - > (batch,seq_len,d_model)

        # x: (batch,seq_len,d_model) - >  (batch,seq_len,d_model/2,2) - > (batch,seq_len,d_model/2)
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))

        # (batch,seq_len,d_model/2) * (seq_len,d_model/2) - > (batch,seq_len,d_model/2)
        prod = x_complex * self.rpe[: x.shape[1], :]

        # (batch,seq_len,d_model/2) -> (batch,seq_len,d_model/2,2)
        x_rotary_embed = torch.view_as_real(prod)

        # (batch,seq_len,d_model/2,2) > (batch,seq_len,d_model)
        x_real = x_rotary_embed.reshape(*x.shape)
        return x_real


if __name__ == "__main__":
    d_model = 4
    seq_len = 4
    batch_size = 4

    rpe = RotaryPositionalEncoding(d_model=d_model, seq_len=seq_len)
    x = torch.rand(4, 4, 4)
    y = rpe(x)
    print(y)
    print(y.shape)
