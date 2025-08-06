import torch
from utils.cal import ContrastiveLoss
from backbone.Resnet import ResNet50Embedding
from torch.utils.data import DataLoader
from torchvision import transforms
import config
import torch.multiprocessing
from dataset.dataset import Market1501Dataset


transform = transforms.Compose([
    transforms.Resize(config.INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    ),
])

def train():

    train_dataset = Market1501Dataset("datasets/archive/Market-1501-v15.09.15/bounding_box_train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = config.NUM_EPOCHS

    model = ResNet50Embedding().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        import tqdm

        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")


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
    torch.multiprocessing.freeze_support()  # 👈 cần cho Windows khi freeze app
    train()