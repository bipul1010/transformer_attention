from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE
from tokenizers.trainers import WordLevelTrainer, BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import random_split, DataLoader
from dataset import BilingualDataset


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(ds, lang, config):
    tokenizer_path = Path(config["base_path"]) / config["tokenizer_file"].format(lang)
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=5
        )
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataset(config):
    ds_raw = load_dataset(
        f"{config['datasource']}",
        f"{config['lang_src']}-{config['lang_tgt']}",
        split="train",
    )

    tokenizer_src = get_or_build_tokenizer(ds_raw, config["lang_src"], config)
    tokenizer_tgt = get_or_build_tokenizer(ds_raw, config["lang_tgt"], config)

    ##finding max sequence length
    # max_src_length = 0
    # max_tgt_length = 0
    # for item in ds_raw:
    #     src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
    #     tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids

    #     max_src_length = max(len(src_ids), max_src_length)
    #     max_tgt_length = max(len(tgt_ids), max_tgt_length)

    # print(f" Max Source Length: {max_src_length} | Max Tgt Length: {max_tgt_length}")

    # print(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())

    train_ds_size = int(len(ds_raw) * 0.9)
    print(train_ds_size)
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["seq_len"],
        config["lang_src"],
        config["lang_tgt"],
    )

    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["seq_len"],
        config["lang_src"],
        config["lang_tgt"],
    )

    train_dataloader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


if __name__ == "__main__":
    from config import get_config

    config = get_config()
    toke = get_dataset(config)

    print(toke[2].get_vocab_size(), toke[3].get_vocab_size())

    for batch in toke[0]:
        print(batch["encoder_padding_mask"].shape)
        print(batch["decoder_padding_mask"].shape)
        break
    # x = toke.encode("How are you?")
    # print(x.ids)
    # print(toke.decode(x.ids))

    # sos_token = torch.tensor([toke.token_to_id("[PAD]")], dtype=torch.int64)

    # print(torch.cat([torch.tensor([12, 13]), sos_token]))
