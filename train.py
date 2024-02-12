import torch
import sys
import os
from tqdm import tqdm
from torch.nn import functional as F
from pathlib import Path

import wandb
from dataset_tokenizer import get_dataset
from model import build_transformer
from validation import check_validation_result
from config import get_weights_file_path, latest_weights_file_path
from dataset import generate_square_subsequent_mask
import argparse

##Distributed Training
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist


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

    assert device == "cuda", "Training on GPU is not supported."
    print(f"Using Device: {device}")
    device = torch.device(device)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    src_vocab_size, tgt_vocab_size = (
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
    )
    model = get_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        config=config,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    wandb_run_id = None

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
        print(f"GPU Local Rank: {config['local_rank']}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        wandb_run_id = state["wandb_run_id"]
        del state
    else:
        ##if we couldn't find a model - just start it from scratch
        print(
            f" GPU Local Rank: {config['local_rank']} - ModelFilename: {model_filename} not present - not loaded any pretrained model"
        )
    ##only initialize W&B on the global rank 0 mode
    if config["global_rank"] == 0:
        wandb.init(
            project="Distributed Training",
            id=wandb_run_id,
            resume="allow",
            config=config,
        )

        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="global_step")

    attn_mask = generate_square_subsequent_mask(size=config["seq_len"]).to(
        device
    )  ## (seq_length,seq_length)

    ##convert the model to DistributedParallel
    model = DistributedDataParallel(model, device_ids=[config["local_rank"]])

    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(
            train_dataloader,
            desc=f"Processing Epocs: {epoch} on global rank: {config['global_rank']} and local rank: {config['local_rank']}",
        )
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
            encoder_output = model.module.encode(
                x=encoder_input, src_mask=encoder_mask
            )  ## (batch,seq_length,d_model)
            decoder_output = model.module.decode(
                x=decoder_input,
                encoder_output=encoder_output,
                src_mask=encoder_mask,
                tgt_mask=decoder_mask,
                attn_mask=attn_mask,
            )  ## (batch,seq_length,d_model)

            logits = model.module.projection(
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

            if config["global_rank"] == 0:
                wandb.log({"train/loss": loss.item(), "global_step": global_step})

            ##back propagate the loss
            loss.backward()

            ##update the weights
            optimizer.step()

            ##set the gradient to be zero
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        batch_iterator.write(
            f"Epoch: {epoch} | Training Loss: {sum(losses)/len(losses)}"
        )

        if config["global_rank"] == 0:
            wandb.log({"train/avgloss": sum(losses) / len(losses), "epch": epoch})
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
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "wandb_run_id": wandb.run.id,
                    },
                    model_filename,
                )


if __name__ == "__main__":
    from config import get_config

    config = get_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default=config["base_path"])

    args = parser.parse_args()

    config.update(vars(args))

    config["local_rank"] = int(os.environ["LOCAL_RANK"])
    config["global_rank"] = int(os.environ["RANK"])
    # print(config)
    # sys.exit()

    assert config["local_rank"] != -1, "Local Rank can't be <0"
    init_process_group(backend="gloo")
    torch.cuda.set_device(config["local_rank"])

    train(config)
    destroy_process_group()
