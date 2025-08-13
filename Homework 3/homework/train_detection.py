import torch
import torchvision
import numpy as np
import cv2

from torch.utils.tensorboard import SummaryWriter
from fire import Fire
from datetime import datetime
from pathlib import Path
from models import Detector, save_model
from datasets.road_dataset import load_data
from metrics import ConfusionMatrix, DetectionMetric, AccuracyMetric
from iou_visualizer import handle_bboxes

def train(
        num_epoch: int = 50,
        lr: float = 1e-4,
        batch_size: int = 128,
        seed: int = 2024,
        exp_dir: str = "logs",
        model_name: str = "detector"
):
    device = torch.device("cuda")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / \
        f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = SummaryWriter(log_dir)

    model = Detector()
    model = model.to(device)
    model.train()

    data_dir = Path(
        "/home/chom/Dropbox/Documents/MSAI/Deep-Learning/Homework 3/drive_data")
    train_data = load_data(data_dir / "train", shuffle=True)
    val_data = load_data(data_dir / "val")

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_func = torch.nn.CrossEntropyLoss()
    depth_loss_func = torch.nn.L1Loss()
    depth_loss_weight = 0.5  # Adjust this weight based on your validation performance
    global_step = 0
    confusion_matrix = ConfusionMatrix(num_classes=3)
    detection_metric = DetectionMetric(num_classes=3)

    for epoch in range(num_epoch):
        model.train()
        train_accuracy, train_miou, train_mae, train_tp_mae = [], [], [], []

        for batch in train_data:

            data = batch["image"].to(device)
            label = batch["track"].to(device)
            depth = batch["depth"].to(device)

            optim.zero_grad()
            logits, pred_depth = model(data)

            seg_loss = loss_func(logits, label)
            depth_loss = depth_loss_func(pred_depth, depth)
            loss = seg_loss + depth_loss_weight * depth_loss

            loss.backward()
            optim.step()

            pred_classes = logits.argmax(dim=1)

            train_accuracy.append(
                (pred_classes == label).float().mean().item())

            detection_metric.add(pred_classes, label, pred_depth, depth)

        metrics = detection_metric.compute()
        train_miou.append(metrics["iou"])
        train_mae.append(metrics["abs_depth_error"])
        train_tp_mae.append(metrics["tp_depth_error"])

        logger.add_scalar("train/accuracy", np.mean(train_accuracy), epoch)
        logger.add_scalar("train/mIoU", np.mean(train_miou), epoch)
        logger.add_scalar("train/MAE", np.mean(train_mae), epoch)
        logger.add_scalar("train/TP_MAE", np.mean(train_tp_mae), epoch)
        logger.add_scalar("train/depth_loss", depth_loss.item(), epoch)

        model.eval()
        detection_metric.reset()

        valid_accuracy, valid_miou, valid_mae, valid_tp_mae = [], [], [], []
        for batch in train_data:
            data = batch["image"].to(device)
            label = batch["track"].to(device)
            depth = batch["depth"].to(device)

            with torch.inference_mode():
                logits, pred_depth = model(data)

            pred_classes = logits.argmax(dim=1)

            valid_accuracy.append(
                (pred_classes == label).float().mean().item())

            detection_metric.add(pred_classes, label, pred_depth, depth)

        metrics = detection_metric.compute()
        valid_miou.append(metrics["iou"])
        valid_mae.append(metrics["abs_depth_error"])
        valid_tp_mae.append(metrics["tp_depth_error"])

        logger.add_scalar("valid/accuracy", np.mean(valid_accuracy), epoch)
        logger.add_scalar("valid/mIoU", np.mean(valid_miou), epoch)
        logger.add_scalar("valid/MAE", np.mean(valid_mae), epoch)
        logger.add_scalar("valid/TP_MAE", np.mean(valid_tp_mae), epoch)

        logger.flush()

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={np.mean(train_accuracy):.4f}, val_acc={np.mean(valid_accuracy):.4f}, "
                f"train_mIoU={np.mean(train_miou):.4f}, val_mIoU={np.mean(valid_miou):.4f}, "
                f"train_MAE={np.mean(train_mae):.4f}, val_MAE={np.mean(valid_mae):.4f}, "
                f"train_TP_MAE={np.mean(train_tp_mae):.4f}, val_TP_MAE={np.mean(valid_tp_mae):.4f}"
            )
    save_model(model)

    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    Fire(train)
