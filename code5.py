import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Data
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

# 2. Model
def create_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

# 3. Training function
def train_model(optimizer_name):
    model = create_model()
    criterion = nn.CrossEntropyLoss()

    # Choose optimizer
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.01)

    elif optimizer_name == "Momentum":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_losses = []

    # Training loop
    for epoch in range(5):
        total_train_loss = 0

        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_losses.append(total_train_loss)

        # Testing
        total_test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        test_losses.append(total_test_loss)

        print(f"{optimizer_name} Epoch {epoch+1}, Train Loss: {total_train_loss:.4f}, Test Acc: {100*correct/total:.2f}%")

    return train_losses, test_losses

# 4. Run all optimizers
optimizers = ["SGD", "Momentum", "RMSprop", "Adam"]

results = {}

for opt in optimizers:
    train_l, test_l = train_model(opt)
    results[opt] = (train_l, test_l)

# 5. Plot
for opt in results:
    plt.plot(results[opt][0], label=f"{opt} Train")
    plt.plot(results[opt][1], label=f"{opt} Test")

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Optimizer Comparison")
plt.show()
