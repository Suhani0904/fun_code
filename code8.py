import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -----------------------
# DATA
# -----------------------
transform = transforms.ToTensor()

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

# -----------------------
# LENET MODEL
# -----------------------
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)   # (1,28,28) → (6,24,24)
        self.pool = nn.MaxPool2d(2)       # → (6,12,12)
        self.conv2 = nn.Conv2d(6, 16, 5)  # → (16,8,8)
                                          # pool → (16,4,4)

        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(-1, 16*4*4)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# -----------------------
# TRAIN FUNCTION
# -----------------------
def train_model(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(3):
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# -----------------------
# FEATURE MAP HOOK
# -----------------------
feature_maps = []

def hook_fn(module, input, output):
    feature_maps.append(output)

# -----------------------
# RUN
# -----------------------
model = LeNet()

# register hook on conv layers
model.conv1.register_forward_hook(hook_fn)
model.conv2.register_forward_hook(hook_fn)

train_model(model)

# -----------------------
# VISUALIZATION
# -----------------------
model.eval()

images, _ = next(iter(test_loader))
output = model(images)

# plot feature maps
for i, fmap in enumerate(feature_maps):
    fmap = fmap.detach()

    num_maps = fmap.shape[1]

    plt.figure(figsize=(10,5))
    for j in range(min(num_maps, 6)):
        plt.subplot(1,6,j+1)
        plt.imshow(fmap[0, j].cpu(), cmap='gray')
        plt.axis('off')
    plt.suptitle(f"Feature Maps from layer {i+1}")
    plt.show()
