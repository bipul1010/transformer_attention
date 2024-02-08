import torch
from torch._C import device
import torch.nn as nn
from pathlib import Path
from tokenizers import Tokenizer
from train import get_model
from config import get_weights_file_path, latest_weights_file_path
from dataset import generate_square_subsequent_mask


def load_model(model, config, epoch_index: int):
    if epoch_index >= 0:
        model_file_path = get_weights_file_path(config=config, epoch=epoch_index)
    else:
        model_file_path = latest_weights_file_path(config=config)

    if Path(model_file_path).exists() is False:
        raise FileNotFoundError(f"Modelfilename not found:{model_file_path}")

    state = torch.load(model_file_path)
    model.load_state_dict(state["model_state_dict"])

    return model


def load_tokenizers(config):
    tokenizer_src_file = config["tokenizer_file"].format(config["lang_src"])
    tokenizer_tgt_file = config["tokenizer_file"].format(config["lang_tgt"])

    tokenizer_base_path = Path(f"{config['base_path']}")
    tokenizer_src = Tokenizer.from_file(str(tokenizer_base_path / tokenizer_src_file))
    tokenizer_tgt = Tokenizer.from_file(str(tokenizer_base_path / tokenizer_tgt_file))
    return tokenizer_src, tokenizer_tgt


def load_inference_args(config, epoch_index=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_src, tokenizer_tgt = load_tokenizers(config)
    model = get_model(
        src_vocab_size=tokenizer_src.get_vocab_size(),
        tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
        config=config,
        device=device,
    )

    model = load_model(model, config, epoch_index=epoch_index)
    return tokenizer_src, tokenizer_tgt, model


def greedy_output(encoder_input, model, src_mask, tokenizer_tgt, max_length, device):

    decoder_input = (
        torch.tensor([tokenizer_tgt.token_to_id("[SOS]")]).unsqueeze(0).to(device)
    )  ##(1,seq_length)
    encoder_output = model.encode(encoder_input, src_mask)  ## (1,seq_length,d_model)

    query_input = decoder_input.clone()
    start_pos = 0
    while True:

        if decoder_input.shape[1] == max_length:
            break

        attn_mask = generate_square_subsequent_mask(decoder_input.shape[1]).to(
            device
        )  ##(seq_length,seq_length)

        decoder_output = model.decode(
            x=decoder_input,
            encoder_output=encoder_output,
            src_mask=src_mask,
            tgt_mask=None,
            attn_mask=attn_mask,
            start_pos=start_pos,
        )  ##(1,seq_length,d_model)
        projection_output = model.projection(
            decoder_output
        )  ## (1,seq_length,d_model) -> (1,seq_length,vocab_size)

        ##get next token
        prob_layer = projection_output[:, -1, :]
        value, next_token = torch.max(prob_layer, dim=1)

        next_token_id = next_token[-1]
        query_input = torch.tensor([next_token_id]).unsqueeze(0).to(device)
        decoder_input = torch.cat(
            [decoder_input, torch.tensor([next_token_id]).unsqueeze(0).to(device)],
            dim=1,
        )
        if next_token_id == torch.tensor(tokenizer_tgt.token_to_id("[EOS]")):
            break
        start_pos += 1
    return decoder_input


def inference(
    input_text: str, tokenizer_src, tokenizer_tgt, model, config, max_tokens=350
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")
    device = torch.device(device)

    model.eval()
    model = model.to(device)

    sos_token_id = tokenizer_src.token_to_id("[SOS]")
    eos_token_id = tokenizer_src.token_to_id("[EOS]")
    pad_token_id = tokenizer_src.token_to_id("[PAD]")

    encoder_input_tokens = tokenizer_src.encode(input_text).ids
    encoder_padding_length = config["seq_len"] - len(encoder_input_tokens) - 2

    encoder_input = (
        torch.cat(
            [
                torch.tensor([sos_token_id]),
                torch.tensor(encoder_input_tokens),
                torch.tensor([eos_token_id]),
                torch.tensor([pad_token_id] * encoder_padding_length),
            ],
            dim=0,
        )
        .unsqueeze(0)
        .to(device)
    )  ##(1,seq_len)
    encoder_mask = encoder_input == pad_token_id  ##(1,seq_len)

    generated_tokens = greedy_output(
        encoder_input=encoder_input,
        model=model,
        src_mask=encoder_mask,
        tokenizer_tgt=tokenizer_tgt,
        max_length=max_tokens,
        device=device,
    )

    predicted_text = tokenizer_tgt.decode(
        generated_tokens.squeeze(0).detach().cpu().tolist()
    )
    ### print text
    print(f"-" * 30)
    print(f"Input Text: {input_text}")
    print(f"Predicted Text: {predicted_text}")

    return predicted_text


if __name__ == "__main__":
    from config import get_config

    config = get_config()
    tokenizer_src, tokenizer_tgt, model = load_inference_args(config)

    x = inference(
        "Here, Tanya! That's for you!...",
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        model=model,
        config=config,
    )
