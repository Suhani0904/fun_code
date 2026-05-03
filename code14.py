import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------
# DATA
# -----------------------
transform = transforms.ToTensor()

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# -----------------------
# RBM MODEL
# -----------------------
class RBM(nn.Module):
    def __init__(self, visible, hidden):
        super().__init__()

        self.W = nn.Parameter(torch.randn(hidden, visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(hidden))
        self.v_bias = nn.Parameter(torch.zeros(visible))

    def sample_h(self, v):
        prob = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
        return prob, torch.bernoulli(prob)

    def sample_v(self, h):
        prob = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
        return prob, torch.bernoulli(prob)

    def forward(self, v):
        prob_h, h = self.sample_h(v)
        prob_v, v = self.sample_v(h)
        return v

# -----------------------
# TRAIN RBM (CD-1)
# -----------------------
def train_rbm(rbm, loader, epochs=5, lr=0.01):
    optimizer = optim.SGD(rbm.parameters(), lr=lr)

    for epoch in range(epochs):
        loss_total = 0

        for images, _ in loader:
            v0 = images.view(images.size(0), -1)

            prob_h0, h0 = rbm.sample_h(v0)
            prob_v1, v1 = rbm.sample_v(h0)
            prob_h1, h1 = rbm.sample_h(v1)

            loss = torch.mean((v0 - v1) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()

        print(f"RBM Epoch {epoch+1}, Loss: {loss_total/len(loader)}")

    return rbm

# -----------------------
# TRAIN TWO RBMs
# -----------------------
rbm1 = RBM(784, 256)
rbm1 = train_rbm(rbm1, train_loader)

# Transform data through RBM1
def transform_data(rbm, loader):
    data = []
    for images, labels in loader:
        v = images.view(images.size(0), -1)
        prob_h, _ = rbm.sample_h(v)
        data.append((prob_h.detach(), labels))
    return data

rbm1_data = transform_data(rbm1, train_loader)

# Train second RBM
rbm2 = RBM(256, 128)
optimizer = optim.SGD(rbm2.parameters(), lr=0.01)

for epoch in range(5):
    total_loss = 0
    for v, _ in rbm1_data:
        prob_h, h = rbm2.sample_h(v)
        prob_v, v1 = rbm2.sample_v(h)
        loss = torch.mean((v - v1) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"RBM2 Epoch {epoch+1}, Loss: {total_loss/len(rbm1_data)}")

# -----------------------
# DBN CLASSIFIER
# -----------------------
class DBN(nn.Module):
    def __init__(self, rbm1, rbm2):
        super().__init__()

        self.rbm1 = rbm1
        self.rbm2 = rbm2

        # freeze RBMs
        for param in self.rbm1.parameters():
            param.requires_grad = False
        for param in self.rbm2.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x, _ = self.rbm1.sample_h(x)
        x, _ = self.rbm2.sample_h(x)
        x = self.fc(x)
        return x

# -----------------------
# TRAIN DBN CLASSIFIER
# -----------------------
dbn = DBN(rbm1, rbm2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(dbn.parameters(), lr=0.001)

for epoch in range(5):
    total_loss = 0
    for images, labels in train_loader:
        outputs = dbn(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"DBN Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

# -----------------------
# MLP FOR COMPARISON
# -----------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

mlp = MLP()
optimizer = optim.Adam(mlp.parameters(), lr=0.001)

for epoch in range(5):
    for images, labels in train_loader:
        outputs = mlp(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# -----------------------
# EVALUATION
# -----------------------
def evaluate(model):
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

print("DBN Accuracy:", evaluate(dbn))
print("MLP Accuracy:", evaluate(mlp))
