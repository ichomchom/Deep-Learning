"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
import argparse

from models import MLPPlanner, TransformerPlanner, CNNPlanner, save_model
from datasets.road_dataset import load_data

print("Time to train")


def train(
        num_epoch: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
        seed: int = 42,
        exp_dir: str = "logs",
        model_name: str = "transformer_planner",
        d_model: int = 256,
        n_heads: int = 8,
        ** kwargs,
):
    device = torch.device("cuda")
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / \
        f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = SummaryWriter(log_dir)

    model = TransformerPlanner()
    model = model.to(device)
    model.train()
    data_dir = Path(
        "/home/chom/Dropbox/Documents/MSAI/Deep-Learning/Homework 4/drive_data")
    train_data = load_data(data_dir / "train", shuffle=True)
    val_data = load_data(data_dir / "val")

    loss_func = torch.nn.L1Loss()
    optim = torch.optim.AdamW(model.parameters(), lr)

    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0

        for batch in train_data:
            left = batch["track_left"].to(device)
            right = batch["track_right"].to(device)
            target = batch["waypoints"].to(device)

            optim.zero_grad()
            output = model(track_left=left, track_right=right)
            # output = model(image=batch["image"].to(device))
            loss = loss_func(output, target)
            loss.backward()
            optim.step()

            train_loss += loss.item()

        train_loss /= len(train_data)

        model.eval()
        val_loss = 0.0
        total_long_error, total_lat_error = 0.0, 0.0
        count = 0
        with torch.no_grad():
            for batch in val_data:
                left = batch["track_left"].to(device)
                right = batch["track_right"].to(device)
                target = batch["waypoints"].to(device)
                mask = batch["waypoints_mask"].to(device)

                output = model(track_left=left, track_right=right)
                # output = model(image=batch["image"].to(device))
                loss = loss_func(output, target)
                val_loss += loss.item()

                output_masked = output[mask]
                target_masked = target[mask]
                diff = torch.abs(output_masked - target_masked)

                long_error = diff[:, 0].mean().item()
                lat_error = diff[:, 1].mean().item()
                total_lat_error += lat_error
                total_long_error += long_error
                count += 1

        val_loss /= len(val_data)
        avg_lat_error = total_lat_error / count
        avg_long_error = total_long_error / count

        logger.add_scalar("train_loss", train_loss, epoch)
        logger.add_scalar("val_loss", val_loss, epoch)
        logger.add_scalar("val_lat_error", avg_lat_error, epoch)
        logger.add_scalar("val_long_error", avg_long_error, epoch)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:02d}/{num_epoch}: "
                f"Train Loss={train_loss:.4f}, "
                f"Val Loss={val_loss:.4f}, "
                f"Lat Error={avg_lat_error:.4f}, "
                f"Long Error={avg_long_error:.4f}"
            )

    save_model(model)
    torch.save(model.state_dict, log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir} / {model_name}.th")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    train(**vars(parser.parse_args()))
