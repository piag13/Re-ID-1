import torch

# ================================
# 🚀 Device Configuration
# ================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# 🏷️ Training Hyperparameters
# ================================
BATCH_SIZE = 32          # Nếu có batch_size, bạn có thể chỉnh ở đây
NUM_EPOCHS = 10          # Số epoch
LEARNING_RATE = 0.0001   # Learning rate
WEIGHT_DECAY = 0.0005    # Nếu có dùng regularization
MOMENTUM = 0.9           # Cho SGD optimizer (nếu dùng)

# ================================
# 📂 Dataset Paths
# ================================
DATASET_PATH = "datasets/archive/Market-1501-v15.09.15/bounding_box_train"
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"

# ================================
# 🎯 Model Parameters
# ================================
INPUT_SIZE = (224, 224)  # Resize cho ResNet50
BACKBONE = "resnet50"    # resnet50 / facenet / efficientnet
PRETRAINED = True        # Có dùng pretrained weight không

# ================================
# 🌱 Reproducibility
# ================================
SEED = 42
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
