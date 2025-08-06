import random
from PIL import Image
import os
from torch.utils.data import Dataset

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