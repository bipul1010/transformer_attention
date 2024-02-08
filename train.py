import torch
import sys
from torch._C import device
from tqdm import tqdm
from torch.nn import functional as F
from pathlib import Path
from dataset_tokenizer import get_dataset
from model import build_transformer
from validation import check_validation_result
from config import get_weights_file_path, latest_weights_file_path
from dataset import generate_square_subsequent_mask


def get_model(src_vocab_size: int, tgt_vocab_size: int, config, device):
    model = build_transformer(
        d_model=config["d_model"],
        h=config["no_heads"],
        seq_length=config["seq_len"],
        N=config["no_of_layers"],
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dropout=config["dropout"],
        batch_size=config["batch_size"],
        device=device,
    )
    return model


def train(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")
    device = torch.device(device)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    src_vocab_size, tgt_vocab_size = (
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
    )
    model = get_model(
        src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, config=config
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    folder_path = (
        f"{config['base_path']}/{config['datasource']}_{config['model_folder']}"
    )
    if Path(folder_path).exists() is False:
        Path(folder_path).mkdir()
    # sys.exit()
    ##loading any pretrained model for further training if present.
    preloaded_model = config["preload"]
    model_filename = (
        latest_weights_file_path(config) if preloaded_model == "latest" else None
    )
    if model_filename:
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]
    else:
        print(
            f" ModelFilename: {model_filename} not present - not loaded any pretrained model"
        )
    attn_mask = generate_square_subsequent_mask(size=config["seq_len"]).to(
        device
    )  ## (seq_length,seq_length)
    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epocs: {epoch}")
        losses = []
        # batch_index = 0
        for training_batch in batch_iterator:
            # batch_index += 1
            # if batch_index >= 10:
            #     break
            encoder_input = training_batch["encorder_input"].to(
                device
            )  ## (batch,seq_length)
            decoder_input = training_batch["decorder_input"].to(
                device
            )  ## (batch, seq_length)
            label = training_batch["label"].to(device)  ## (batch,seq_length)
            encoder_mask = training_batch["encoder_padding_mask"].to(
                device
            )  ## (batch,seq_length)
            decoder_mask = training_batch["decoder_padding_mask"].to(
                device
            )  ## (batch,seq_length)

            ## model - encoding - > decoding ->  projection_layer
            encoder_output = model.encode(
                x=encoder_input, src_mask=encoder_mask
            )  ## (batch,seq_length,d_model)
            decoder_output = model.decode(
                x=decoder_input,
                encoder_output=encoder_output,
                src_mask=encoder_mask,
                tgt_mask=decoder_mask,
                attn_mask=attn_mask,
            )  ## (batch,seq_length,d_model)

            logits = model.projection(
                x=decoder_output
            )  ## (batch,seq_length,d_model) - > (batch,seq_length,vocab_size)

            ##computing loss
            B, S, V = logits.shape
            logits = logits.view(B * S, V)
            label = label.view(B * S)
            loss = F.cross_entropy(
                logits,
                label,
                label_smoothing=0.1,
                ignore_index=tokenizer_src.token_to_id("[PAD]"),
            )
            batch_iterator.set_postfix({"loss": loss.item()})

            losses.append(loss.item())

            optimizer.zero_grad(set_to_none=True)

            ##back propagate the loss
            loss.backward()

            ##update the weights
            optimizer.step()

            global_step += 1

        batch_iterator.write(
            f"Epoch: {epoch} | Training Loss: {sum(losses)/len(losses)}"
        )
        ##Run validation
        check_validation_result(
            val_dataloader=val_dataloader,
            model=model,
            tokenizer_tgt=tokenizer_tgt,
            print_msg=lambda msg: batch_iterator.write(msg),
            num_examples=1,
            device=device,
            config=config,
        )

        ##saving model at the end of each epoch
        if epoch % 4 == 0:
            model_filename = get_weights_file_path(config, epoch=epoch)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                },
                model_filename,
            )


if __name__ == "__main__":
    from config import get_config

    config = get_config()
    train(config)
