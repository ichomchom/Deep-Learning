import torch
import torchvision
import numpy as np
import cv2

from torch.utils.tensorboard import SummaryWriter
from fire import Fire
from datetime import datetime
from pathlib import Path
from models import Classifier, save_model
from datasets.classification_dataset import load_data


def train(
        num_epoch: int = 50,
        lr: float = 1e-3,
        batch_size: int = 128,
        seed: int = 2024,
        exp_dir: str = "logs",
        model_name: str = "classifier"
):
    device = torch.device("cuda")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / \
        f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = SummaryWriter(log_dir)

    model = Classifier()
    model = model.to(device)
    model.train()

    data_dir = Path(
        "/home/chom/Dropbox/Documents/MSAI/Deep-Learning/Homework 3/classification_data")
    train_data = load_data(data_dir / "train", shuffle=True)
    val_data = load_data(data_dir / "val")

    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()
    global_step = 0

    for epoch in range(num_epoch):
        model.train()
        train_accuracy = []
        for data, label in train_data:
            data, label = data.to(device), label.to(device)

            output = model(data)
            loss = loss_func(output, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            accuracy = (output.argmax(dim=-1) ==
                        label).cpu().detach().float().numpy()

            train_accuracy.extend(
                accuracy
            )

        logger.add_scalar("train/accuracy", np.mean(train_accuracy), epoch)

        model.eval()
        valid_accuracy = []
        for data, label in val_data:
            data, label = data.to(device), label.to(device)

            with torch.inference_mode():
                output = model(data)

            valid_accuracy.extend(
                (output.argmax(dim=-1) == label).cpu().detach().float().numpy()
            )

        logger.add_scalar("valid/accuracy", np.mean(valid_accuracy), epoch)

        logger.flush()
        epoch_train_acc = torch.as_tensor(train_accuracy).mean().item()
        epoch_val_acc = torch.as_tensor(valid_accuracy).mean().item()

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )
    save_model(model)

    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    Fire(train)
