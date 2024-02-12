from dataset import causal_mask
import torch
import torch.nn.functional as F
from dataset import generate_square_subsequent_mask


def greedy_output(encoder_input, model, src_mask, tokenizer_tgt, max_length, device):

    decoder_input = (
        torch.tensor([tokenizer_tgt.token_to_id("[SOS]")]).unsqueeze(0).to(device)
    )  ##(1,seq_length) -since here batch is 1 for validation
    encoder_output = model.module.encode(
        encoder_input, src_mask
    )  ## (batch,seq_length,d_model)

    while True:

        if decoder_input.shape[1] == max_length:
            break

        attn_mask = generate_square_subsequent_mask(decoder_input.shape[1]).to(device)

        decoder_output = model.module.decode(
            x=decoder_input,
            encoder_output=encoder_output,
            src_mask=src_mask,
            tgt_mask=None,
            attn_mask=attn_mask,
        )  ##(batch,seq_length,d_model)
        projection_output = model.module.projection(
            decoder_output
        )  ## (batch,seq_length,d_model) -> (batch,seq_length,vocab_size)

        ##get next token
        prob_layer = projection_output[:, -1, :]
        value, next_token = torch.max(prob_layer, dim=1)

        next_token_id = next_token[-1]
        decoder_input = torch.cat(
            [decoder_input, torch.tensor([next_token_id]).unsqueeze(0).to(device)],
            dim=1,
        )
        if next_token_id == torch.tensor(tokenizer_tgt.token_to_id("[EOS]")):
            break
    return decoder_input


def check_validation_result(
    val_dataloader, model, tokenizer_tgt, print_msg, num_examples, device, config
):

    with torch.no_grad():
        model.eval()
        count = 0
        for val_batch in val_dataloader:
            encoder_input = val_batch["encorder_input"].to(
                device
            )  ## (batch,seq_length)
            encoder_mask = val_batch["encoder_padding_mask"].to(
                device
            )  ## (batch,seq_length)
            source_text = val_batch["src_text"][0]
            tgt_text = val_batch["tgt_text"][0]

            generated_tokens = greedy_output(
                encoder_input=encoder_input,
                model=model,
                src_mask=encoder_mask,
                tokenizer_tgt=tokenizer_tgt,
                max_length=config["seq_len"],
                device=device,
            )

            predicted_text = tokenizer_tgt.decode(
                generated_tokens.squeeze(0).detach().cpu().tolist()
            )
            ### print text
            print_msg(f"-" * 30)
            print_msg(f"Source Text: {source_text}")
            print_msg(f"Target Text: {tgt_text}")
            print_msg(f"Predicted Text: {predicted_text}")

            count += 1

            if count == num_examples:
                break
