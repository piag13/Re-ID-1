import torch
from utils.cal import ContrastiveLoss
from backbone.Resnet import ResNet50Embedding
import random
from PIL import Image
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import config
import torch.multiprocessing


transform = transforms.Compose([
    transforms.Resize(config.INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    ),
])

class Market1501Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for filename in os.listdir(root_dir):
            if filename.endswith('.jpg'):
                person_id = int(filename.split('_')[0])
                if person_id == -1:  # Skip junk images
                    continue
                path = os.path.join(root_dir, filename)
                self.samples.append((path, person_id))

        # Map person IDs to class indices
        self.id_to_label = {id_: idx for idx, id_ in enumerate(sorted(set(pid for _, pid in self.samples)))}
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}

        # Group indices by label
        self.label_to_indices = {}
        for idx, (_, pid) in enumerate(self.samples):
            label = self.id_to_label[pid]
            self.label_to_indices.setdefault(label, []).append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img1_path, pid1 = self.samples[index]
        label1 = self.id_to_label[pid1]
        img1 = Image.open(img1_path).convert('RGB')

        if random.random() < 0.5:
            # Positive pair
            positive_idx = index
            while positive_idx == index:
                positive_idx = random.choice(self.label_to_indices[label1])
            img2_path, _ = self.samples[positive_idx]
            label = 1
        else:
            # Negative pair
            negative_labels = list(set(self.label_to_indices.keys()) - {label1})
            label2 = random.choice(negative_labels)
            negative_idx = random.choice(self.label_to_indices[label2])
            img2_path, _ = self.samples[negative_idx]
            label = 0

        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2), label

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