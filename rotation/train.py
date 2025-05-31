import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model import HomographyModel
from dataset import HomographyDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = HomographyDataset("annotations.csv", "images/")
loader = DataLoader(dataset, batch_size=16, shuffle=True)


model = HomographyModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(30):
    total_loss = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        preds = model(inputs)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[{epoch+1}] loss: {total_loss / len(loader):.4f}")

torch.save(model.state_dict(), "homography_model.pth")
