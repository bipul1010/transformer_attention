import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from attention import TransformerAttention
from rpe import RotaryPositionalEncoding
from typing import Optional, ValuesView
from kv_cache import KVCache


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(self.vocab_size, self.d_model)

    ## b,seq_len ->  b,seql_len,d_model
    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)


# class PositionEncoding(nn.Module):
#     def __init__(self, seq_length: int, d_model: int, dropout: float) -> None:
#         super().__init__()
#         self.seq_length = seq_length
#         self.d_model = d_model
#         self.dropout = nn.Dropout(dropout)

#         pe = torch.zeros(self.seq_length, self.d_model)
#         pos_vectors = (
#             torch.arange(0, self.seq_length).float().unsqueeze(1)
#         )  ## 1D vector- > (seq_len,1)

#         deno = torch.exp(
#             torch.arange(0, self.d_model, 2).float()
#             * -(math.log(10000.0) / self.d_model)
#         )  ## (d_model/2)

#         pe[:, 0::2] = torch.sin(pos_vectors * deno)  ## sin(pos/10,000**2i/d_model)
#         pe[:, 1::2] = torch.cos(pos_vectors * deno)  ## cos(pos/10,000**2i/d_model)

#         pe = pe.unsqueeze(0)  ## (seq_len,d_model) -> (1,seq_len,d_model)

#         # Register the positional encoding as a buffer
#         self.register_buffer("pe", pe)

#     def forward(self, x):
#         ##(batch,seq_len,d_model) + (1,seq_len,d_model) - > (batch,seq_len,d_model)
#         x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
#         return self.dropout(x)  ## (batch,seq_len,d_model)


class Layernorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):

        ##(batch,seq_length,d_model) -> (batch,seq_length,d_model) ##normalization happened on embedding features.
        return self.norm(x)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        w1 = nn.Linear(d_model, d_ff)
        w2 = nn.Linear(d_ff, d_model)
        relu = nn.ReLU()
        dropout1 = nn.Dropout(dropout)
        dropout2 = nn.Dropout(dropout)

        self.ff = nn.Sequential(w1, relu, dropout1, w2, dropout2)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.ff(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_v = self.d_model // self.h
        self.w_query = nn.Linear(d_model, d_model, bias=False)
        self.w_key = nn.Linear(d_model, d_model, bias=False)
        self.w_value = nn.Linear(d_model, d_model, bias=False)

        self.w_0 = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.final_dropout = nn.Dropout(dropout)

    def self_attention(self, query, key, value, mask):

        ##Input : query,key,value -> (batch,h, seq_len,d_v)

        ##(batch,h, seq_len,d_v) -> (batch,h,seq_len,seq_len)
        attention = torch.matmul(query, key.transpose(-2, -1))
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention.masked_fill_(mask == 0, -1e9)

        attention_score = F.softmax(attention, dim=-1)  ##(batch,h,seq_len,seq_len)
        attention_score = self.dropout(attention_score)  ##(batch,h,seq_len,seq_len)

        ##(batch,h,seq_len,seq_len) @ (batch,h,seq_len,d_v) -> (batch,h,seq_len,d_v)
        output = torch.matmul(attention_score, value)

        return output, attention_score

    def forward(self, q, k, v, mask):
        query = self.w_query(q)
        key = self.w_key(k)
        value = self.w_value(v)
        ## (batch,seq_len,d_model) - > (batch,seq_len,h,d_v) - > (batch,h,seq_len,d_v)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_v).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_v).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_v).transpose(
            1, 2
        )

        x, self.attention_scores = self.self_attention(query, key, value, mask)
        ##(batch,h,seq_len,d_v) -> (batch,seq_len,h,d_v) -> (batch,seq_len,d_model)
        x = x.transpose(2, 1).contiguous().view(x.shape[0], -1, self.h * self.d_v)

        ##(batch,seq_len,d_model) -> (batch,seq_len,d_model)
        return self.final_dropout(self.w_0(x))


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()

        self.proj_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        ## (batch,seq_len,d_model)- > (batch,seq_len,vocab_size)
        return self.proj_layer(x)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention: TransformerAttention,
        feed_forward: FeedForward,
        norm_layers: nn.ModuleList,
        rpe: RotaryPositionalEncoding,
        d_model: int,
    ) -> None:
        super().__init__()
        self.sa = self_attention
        self.ff = feed_forward
        self.norm_layers = norm_layers
        self.rpe = rpe

        self.wq = nn.Linear(d_model, d_model, bias=True)
        self.wk = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x, src_mask):

        ##1st residual connection
        x = self.norm_layers[0](x)

        q = self.rpe(self.wq(x))
        k = self.rpe(self.wk(x))

        x = x + self.sa(q, k, x, key_padding_mask=src_mask, attn_mask=None)

        ##2nd residual connection
        x = self.norm_layers[1](x)
        x = x + self.ff(x)

        return x


class Encoder(nn.Module):
    def __init__(self, encoder_blocks: list) -> None:
        super().__init__()

        self.encoder_blocks = nn.ModuleList(encoder_blocks)

    def forward(self, x, src_mask):
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, src_mask)

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention: TransformerAttention,
        cross_attention: TransformerAttention,
        ff: FeedForward,
        norm_layers: nn.ModuleList,
        rpe: RotaryPositionalEncoding,
        d_model: int,
        kv_cache: KVCache,
    ) -> None:
        super().__init__()

        self.sa = self_attention
        self.ca = cross_attention
        self.ff = ff
        self.norm_layers = norm_layers
        self.rpe = rpe

        self.sa_wq = nn.Linear(d_model, d_model, bias=True)
        self.sa_wk = nn.Linear(d_model, d_model, bias=True)

        self.ca_wq = nn.Linear(d_model, d_model, bias=True)
        self.ca_wk = nn.Linear(d_model, d_model, bias=True)

        self.kv_cache = kv_cache

    def forward(self, x, encoder_output, src_mask, tgt_mask, causal_mask, start_pos):

        ##1st residual connection

        batch_size, seq_len, _ = x.shape

        x = self.norm_layers[0](x)
        query, key, value = x, x, x

        q = self.rpe(self.sa_wq(query))
        k = self.rpe(self.sa_wk(key))
        v = value

        # self.kv_cache.update(xk=k, xv=v, batch_size=batch_size, start_pos=start_pos)

        # keys, values = self.kv_cache.get(batch_size=batch_size, start_pos=start_pos + seq_len)
        x = x + self.sa(q, k, v, key_padding_mask=tgt_mask, attn_mask=causal_mask)

        ##2nd residual connnection

        x = self.norm_layers[1](x)
        ca_q = self.rpe(self.ca_wq(x))
        ca_k = self.rpe(self.ca_wk(encoder_output))

        x = x + self.ca(
            ca_q,
            ca_k,
            encoder_output,
            key_padding_mask=src_mask,
            attn_mask=None,
        )

        ##3rd residual connection
        x = self.norm_layers[2](x)
        x = x + self.ff(x)

        return x


class Decoder(nn.Module):
    def __init__(self, decoder_blocks: list) -> None:
        super().__init__()

        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def forward(self, x, encoder_output, src_mask, tgt_mask, attn_mask, start_pos):
        for decoder_block in self.decoder_blocks:
            x = decoder_block(
                x, encoder_output, src_mask, tgt_mask, attn_mask, start_pos
            )
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_embed: InputEmbedding,
        tgt_embed: InputEmbedding,
        encoder: Encoder,
        decoder: Decoder,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()

        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer

    def encode(self, x, src_mask):
        embed_x = self.src_embed(x)
        return self.encoder(embed_x, src_mask)

    def decode(
        self,
        x,
        encoder_output,
        src_mask,
        tgt_mask,
        attn_mask,
        start_pos: Optional[int] = -1,
    ):
        embed_x = self.tgt_embed(x)
        return self.decoder(
            embed_x, encoder_output, src_mask, tgt_mask, attn_mask, start_pos
        )

    def projection(self, x):
        return self.projection_layer(x)


def build_transformer(
    d_model: int,
    h: int,
    seq_length: int,
    N: int,
    src_vocab_size: int,
    tgt_vocab_size: int,
    dropout: float,
    batch_size: int,
    device: str,
) -> Transformer:

    ##building Input Embeddings and Positional Encoding
    src_embed = InputEmbedding(vocab_size=src_vocab_size, d_model=d_model)
    rpe = RotaryPositionalEncoding(d_model=d_model, seq_len=seq_length)
    kv_cache = KVCache(
        max_batch_size=batch_size,
        max_seq_len=seq_length,
        head_dim=d_model,
        device=device,
    )

    # src_pos_encod = PositionEncoding(
    #     seq_length=seq_length, d_model=d_model, dropout=dropout
    #

    ## Building Encoder Blocks and eventually Encoder
    encoder_blocks = []
    for i in range(N):
        sa = TransformerAttention(d_model=d_model, h=h, dropout=dropout)
        ff = FeedForward(d_model=d_model, d_ff=4 * d_model, dropout=dropout)
        norm_layers = nn.ModuleList([Layernorm(d_model=d_model) for _ in range(2)])
        encoder_block = EncoderBlock(
            self_attention=sa,
            feed_forward=ff,
            norm_layers=norm_layers,
            rpe=rpe,
            d_model=d_model,
        )
        encoder_blocks.append(encoder_block)

    encoder = Encoder(encoder_blocks=encoder_blocks)

    ## Building Output Embeddings and Positional Encoding
    tgt_embed = InputEmbedding(vocab_size=tgt_vocab_size, d_model=d_model)
    # tgt_pos_encod = PositionEncoding(
    #     seq_length=seq_length, d_model=d_model, dropout=dropout
    # )
    # tgt_rpe = RotaryPositionalEncoding(d_model=d_model, seq_len=seq_length)

    ##Building Decoder Blocks and eventually Decoder
    decoder_blocks = []
    for i in range(N):
        sa = TransformerAttention(d_model=d_model, h=h, dropout=dropout)
        ca = TransformerAttention(d_model=d_model, h=h, dropout=dropout)
        ff = FeedForward(d_model=d_model, d_ff=4 * d_model, dropout=dropout)
        norm_layers = nn.ModuleList([Layernorm(d_model=d_model) for _ in range(3)])

        decoder_block = DecoderBlock(
            self_attention=sa,
            cross_attention=ca,
            ff=ff,
            norm_layers=norm_layers,
            rpe=rpe,
            d_model=d_model,
            kv_cache=kv_cache,
        )

        decoder_blocks.append(decoder_block)

    decoder = Decoder(decoder_blocks=decoder_blocks)

    ##Building Projection Layer

    projection_layer = ProjectionLayer(d_model=d_model, vocab_size=tgt_vocab_size)

    ##Finally Building Transformer

    transformer = Transformer(
        src_embed=src_embed,
        tgt_embed=tgt_embed,
        encoder=encoder,
        decoder=decoder,
        projection_layer=projection_layer,
    )

    ##Initializing weights
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    total_parameters = sum([p.numel() for p in transformer.parameters()])
    trainable_parameters = sum(
        [p.numel() for p in transformer.parameters() if p.requires_grad]
    )

    print(
        f"Total Parameters: {total_parameters} | Trainable Parameters: {trainable_parameters}"
    )
    return transformer


if __name__ == "__main__":
    d_model = 512
    seq_length = 8
    dropout = 0.1
    x = InputEmbedding(10, 100)
    t = torch.tensor([[2, 4, 1], [7, 9, 5]])

    transformer = build_transformer(
        d_model=d_model,
        h=8,
        N=6,
        src_vocab_size=1000,
        tgt_vocab_size=2000,
        dropout=0.1,
        seq_length=seq_length,
    )

    print(transformer)
    # embed_t = x(t)
    # print(embed_t, embed_t.shape)

    # pe = PositionEncoding(seq_length=seq_length, d_model=d_model, dropout=0.1)
    # j = pe(embed_t)
    # print(j)
    # print(j.shape)

    # # ff = FeedForward(d_model=d_model, d_ff=4 * d_model, dropout=dropout)

    # # g = ff(j)
    # # print(g.shape)

    # encoder_blocks = []
    # for i in range(6):
    #     ff = FeedForward(d_model=d_model, d_ff=4 * d_model, dropout=dropout)
    #     ma = MultiHeadAttention(d_model=d_model, h=4, dropout=dropout)
    #     # ma_output = ma(j, j, j, mask=None)
    #     # print(ma_output, ma_output.shape)
    #     # ln = LayerNorm(d_model)
    #     norm_layers = nn.ModuleList([Layernorm(d_model=d_model) for _ in range(2)])

    #     encoder_block = EncoderBlock(
    #         self_attention=ma, feed_forward=ff, norm_layers=norm_layers
    #     )
    #     encoder_blocks.append(encoder_block)

    # print()
    # print("Coming here.")
    # encorder = Encoder(encoder_blocks=encoder_blocks)
    # m = encorder(j, None)
    # print(m)
    # print(m.shape)

    # decoder_blocks = []

    # for i in range(6):
    #     sa = MultiHeadAttention(d_model=d_model, h=4, dropout=dropout)
    #     ca = MultiHeadAttention(d_model=d_model, h=4, dropout=dropout)
    #     ff = FeedForward(d_model=d_model, d_ff=4 * d_model, dropout=dropout)
    #     norm_layers = [Layernorm(d_model=d_model) for _ in range(3)]

    #     decoder_block = DecoderBlock(
    #         self_attention=sa, cross_attention=ca, ff=ff, norm_layers=norm_layers
    #     )
    #     decoder_blocks.append(decoder_block)

    # decoder = Decoder(decoder_blocks=decoder_blocks)

    # d = decoder(j, m, None, None)
    # print(d)
    # print(d.shape)
