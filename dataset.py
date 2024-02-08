import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(
        self, ds, tokenizer_src, tokenizer_tgt, seq_length, src_lang, tgt_lang
    ):
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_length = seq_length
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.pad_token_id = [self.tokenizer_src.token_to_id("[PAD]")]
        self.sos_token_id = [self.tokenizer_src.token_to_id("[SOS]")]
        self.eos_token_id = [self.tokenizer_src.token_to_id("[EOS]")]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]

        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        encoder_tokens = self.tokenizer_src.encode(src_text).ids
        decorder_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        encoder_padding_length = (
            self.seq_length - len(encoder_tokens) - 2
        )  ##since we will be adding sos and eos tokens

        decorder_paddin_length = (
            self.seq_length - len(decorder_tokens) - 1
        )  ## since we will be adding only sos in decorder input and eos in label output.

        encorder_input = torch.cat(
            [
                torch.tensor(self.sos_token_id),
                torch.tensor(encoder_tokens),
                torch.tensor(self.eos_token_id),
                torch.tensor(self.pad_token_id * encoder_padding_length),
            ],
            dim=0,
        )

        decorder_input = torch.cat(
            [
                torch.tensor(self.sos_token_id),
                torch.tensor(decorder_tokens),
                torch.tensor(self.pad_token_id * decorder_paddin_length),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(decorder_tokens),
                torch.tensor(self.eos_token_id),
                torch.tensor(self.pad_token_id * decorder_paddin_length),
            ],
            dim=0,
        )

        assert encorder_input.shape[0] == self.seq_length
        assert decorder_input.shape[0] == self.seq_length
        assert label.shape[0] == self.seq_length

        return {
            "encorder_input": encorder_input,  # seq_length
            "decorder_input": decorder_input,  # seq_length
            "label": label,  # seq_length
            "src_text": src_text,
            "tgt_text": tgt_text,
            "encoder_padding_mask": (
                encorder_input == self.pad_token_id[0]
            ),  ## seq_length
            "decoder_padding_mask": (
                decorder_input == self.pad_token_id[0]
            ),  ## seq_length
        }


def causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0


def generate_square_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size) * float("-inf"), diagonal=1)
    return mask


# return {
#     "encorder_input": encorder_input,  # seq_length
#     "decorder_input": decorder_input,  # seq_length
#     "label": label,  # seq_length
#     "src_text": src_text,
#     "tgt_text": tgt_text,
#     "encoder_mask": (encorder_input != self.pad_token_id[0])
#     .int()
#     .unsqueeze(0)
#     .unsqueeze(0),
#     "decorder_mask": (decorder_input != self.pad_token_id[0])  ## (1,1,seq_leng)
#     & causal_mask(decorder_input.shape[0])
#     .int()
#     .unsqueeze(0),  # (1,seq_length, seq_length)
# }
