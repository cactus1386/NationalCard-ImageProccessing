import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load model config
with open("./crnn-base-fa-v2/model_config.yaml", "r", encoding="utf-8") as f:
    model_config = yaml.safe_load(f)

id2label = {int(k): v for k, v in model_config["id2label"].items()}
label2id = {v: k for k, v in id2label.items()}
blank_id = model_config.get("blank_id", 0)

# Load image processor config
with open("./crnn-base-fa-v2/preprocessor/image_processor_config.yaml", "r", encoding="utf-8") as f:
    image_config = yaml.safe_load(f)

image_size = tuple(image_config["size"])
mean = image_config["mean"]
std = image_config["std"]
rescale = image_config.get("rescale", 1.0)
gray_scale = image_config.get("gray_scale", True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset definition
class OCRDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        image = Image.open(img_path)
        if gray_scale:
            image = image.convert("L")
        if self.transform:
            image = self.transform(image)
        with open(label_path, "r", encoding="utf-8") as f:
            label = f.read().strip()
        return image, label

# Label encoding
def encode_label(label):
    return [label2id[c] for c in label if c in label2id]

# Model definition
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        return nn.functional.relu(x)

class CRNN(nn.Module):
    def __init__(self, n_channels, num_classes, map2seq_in_dim, map2seq_out_dim, rnn_dim):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            CNNBlock(n_channels, 64),
            nn.MaxPool2d(2, 2),
            CNNBlock(64, 128),
            nn.MaxPool2d(2, 2),
            CNNBlock(128, 256),
            CNNBlock(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),
            CNNBlock(256, 512),
            nn.MaxPool2d((2, 1), (2, 1)),
            CNNBlock(512, 512),
            CNNBlock(512, 512),
        )
        self.map2seq = nn.Linear(map2seq_in_dim, map2seq_out_dim)
        self.rnn1 = nn.LSTM(map2seq_out_dim, rnn_dim, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(rnn_dim * 2, rnn_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(rnn_dim * 2, num_classes)

    def forward(self, x):
        batch, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).reshape(batch, width, -1)
        x = self.map2seq(x)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.fc(x)
        return x

# Training setup
def train_model(model, dataloader, criterion, optimizer, num_epochs=100):
    best_loss = float("inf")
    patience = 3
    wait = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            targets = [encode_label(l) for l in labels]
            targets_flat = [i for sub in targets for i in sub]
            targets_tensor = torch.tensor(targets_flat, dtype=torch.long).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.log_softmax(2)
            input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long)
            target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
            loss = criterion(outputs, targets_tensor, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            torch.save(model.state_dict(), "./best_crnn_model.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping...")
                break

# Main
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = OCRDataset("./dataset/images_aug", "./dataset/labels_aug", transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    model = CRNN(
        n_channels=model_config["n_channels"],
        num_classes=len(label2id),
        map2seq_in_dim=model_config["map2seq_in_dim"],
        map2seq_out_dim=model_config["map2seq_out_dim"],
        rnn_dim=model_config["rnn_dim"]
    ).to(device)

    # Load pretrained model weights
    if os.path.exists("./crnn-base-fa-v2/model.pt"):
        pretrained_path = "./crnn-base-fa-v2/model.pt"
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        model_dict = model.state_dict()

        filtered_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.size() == model_dict[k].size()
        }

        print(f"Loading {len(filtered_dict)} / {len(model_dict)} layers from pre-trained weights.")

        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        
    criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.00023)

    train_model(model, train_loader, criterion, optimizer)
