# -*- coding: utf-8 -*-
import time
import logging

import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.tree import Tree
import cv2

import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from modules.backbones.resnet import resnet18
from modules.head.classification_head import FcClassificationHead
from agents.loggers.base import TrainLogger


def accuracy(output, target, topk=(1,)):
    """计算 top-k 准确率"""
    maxk = max(topk)
    batch_size = target.size(0)

    # 获取前 maxk 个预测
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    logger = logging.getLogger("PancakeDetector")
    logger.setLevel(logging.INFO)
    logger.addHandler(RichHandler())
    console = Console()
    console.print()

    backbone = resnet18(input_channels=1)
    head = FcClassificationHead(input_channels=512, num_classes=10)
    model = nn.Sequential(backbone, head)

    model_card = Tree("Classification Model")
    backbone_card = model_card.add("Backbone")
    head_card = model_card.add("Head")

    params = sum(p.numel() for p in backbone.parameters())
    backbone_card.add(f"Model Name: {getattr(backbone, 'model_name', backbone.__class__.__name__)}")
    backbone_card.add(f"Parameters: {(params / 1e6):.2f}M, ({params})")

    params = sum(p.numel() for p in head.parameters())
    head_card.add(f"Model Name: {getattr(head, 'model_name', head.__class__.__name__)}")
    head_card.add(f"Parameters: {(params / 1e6):.2f}M, ({params})")

    console.print(
        Panel(
            model_card,
            title="Model Card",
            subtitle="Created with :pancakes:PancakeDetector library:smiling_face_with_3_hearts:",
        )
    )

    # ======================================================================================== #

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev_logger = TrainLogger(
        max_epochs=2,
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    # 定义转换操作
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
        # transforms.Normalize((0.1307,), (0.3081,))  # 标准化
    ])

    train_dataset = MNIST(root=r'./datasets/download', train=True, download=True, transform=transform)
    valid_dataset = MNIST(root=r'./datasets/download', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)

    backbone.train()
    head.train()

    try:
        for epoch in range(2):
            print(f"Epoch: {epoch}")
            dev_logger.on_epoch_start(
                epoch=epoch,
                train_batches=len(train_loader),
                valid_batches=len(valid_loader),
                start_time=time.time(),
            )
            backbone.train()
            head.train()

            for batch_idx, batch_data in enumerate(train_loader):
                dev_logger.on_batch_start(
                    batch=batch_idx,
                    epoch=epoch,
                    start_time=time.time(),
                    batch_size=train_loader.batch_size,
                )
                data: torch.Tensor
                target: torch.Tensor
                data, target = batch_data
                data, target = data.to(device), target.to(device)

                forward = backbone(data)
                forward = head(forward)

                loss = loss_fn(forward, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")

                if batch_idx > 40:
                    break

            backbone.eval()
            head.eval()

            with torch.no_grad():
                top1_acc = 0
                top5_acc = 0
                total_samples = 0

                for batch_idx, batch_data in enumerate(valid_loader):
                    data, target = batch_data
                    data, target = data.to(device), target.to(device)

                    forward = backbone(data)
                    forward = head(forward)

                    loss = loss_fn(forward, target)

                    # 计算准确率
                    top1, top5 = accuracy(forward, target, topk=(1, 5))
                    top1_acc += top1.item() * data.size(0)
                    top5_acc += top5.item() * data.size(0)
                    total_samples += data.size(0)

                    print(f"Batch: {batch_idx}/{len(valid_loader)}, Loss: {loss.item()}, "
                          f"Acc@1: {top1.item():.2f}%, Acc@5: {top5.item():.2f}%")
                    if batch_idx > 10:
                        break

                # 打印最终的准确率
                top1_acc /= total_samples
                top5_acc /= total_samples
                print(f"Epoch: {epoch}, Acc@1: {top1_acc:.2f}%, Acc@5: {top5_acc:.2f}%")

    except KeyboardInterrupt:
        dev_logger.on_early_stopping(reason="KeyboardInterrupt")
        pass

    finally:
        torch.save(model.state_dict(), "model.pth")
