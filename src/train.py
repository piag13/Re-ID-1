import torch
from utils.loss import ContrastiveLoss
from backbone.Resnet import ResNet50Embedding
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.multiprocessing
from dataset.dataset import Market1501Dataset
import yaml
import argparse
import tqdm
import os

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    required_keys = ["dataset_path", "input_size"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")
    if not isinstance(config["input_size"], (list, tuple)) or len(config["input_size"]) != 2:
        raise ValueError("input_size in config must be a tuple or list of length 2")
    return config

def train(epochs, batch_size, learning_rate, device, margin, num_workers):
    train_dataset = Market1501Dataset(config["dataset_path"], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = ResNet50Embedding().to(device)
    criterion = ContrastiveLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for (img1, img2), labels in progress_bar:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device).float()

            emb1 = model(img1)
            emb2 = model(img2)
            loss = criterion(emb1, emb2, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Re-Identification model")
    parser.add_argument('--config', type=str, default="Re-ID-1/src/config/config.yaml", help='Path to configuration file')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Device to use (cuda or cpu)')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for contrastive loss')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loader workers')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command-line arguments if provided
    args.epochs = args.epochs if args.epochs is not None else config.get("num_epochs", 50)
    args.batch_size = args.batch_size if args.batch_size is not None else config.get("batch_size", 32)
    args.lr = args.lr if args.lr is not None else config.get("learning_rate", 0.0001)
    args.device = args.device if args.device is not None else config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(config["input_size"]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        ),
    ])

    torch.multiprocessing.freeze_support()
    train(args.epochs, args.batch_size, args.lr, args.device, args.margin, args.num_workers)