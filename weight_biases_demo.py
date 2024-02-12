import wandb
import random


wandb.init(
    project="hello wandb project!!",
    resume="allow",
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)


wandb.define_metric("global_epoch")
wandb.define_metric("train/*", step_metric="global_epoch")


# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    print(epoch)
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    wandb.log({"train/accu": acc, "global_epoch": epoch})
    wandb.log({"train/loss": loss, "global_epoch": epoch})


wandb.finish()
