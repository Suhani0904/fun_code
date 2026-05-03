import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_data = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
train_loader=DataLoader(train_data, batch_size=64, shuffle=True)
test_loader=DataLoader(test_data, batch_size=64, shuffle=False)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self,x):
        return self.model(x)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(1,16,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )  
        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*5*5,10)
        )  
    def forward(self,x):
        x=self.conv(x )
        x=self.fc(x)
        return x
def count_params(model):
    return sum(p.numel() for p in model.parameters())
def train_model(model):
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    start=time.time()
    for epoch in range(3):
        for images,labels in train_loader:
            outputs=model(images)
            loss=criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    end=time.time()
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for images,labels in test_loader:
            output=model(images)
            _,pred=torch.max(output,1)
            correct+=(pred==labels).sum().item()
            total+=labels.size(0)   
    test_acc=correct/total
    return test_acc,end-start   
mlp=MLP()
cnn=CNN()
mlp_acc, mlp_time = train_model(mlp)
cnn_acc, cnn_time = train_model(cnn)
mlp_params=count_params(mlp)
cnn_params=count_params(cnn)
print(f"MLP - Params: {mlp_params}, Accuracy: {mlp_acc:.4f}, Time: {mlp_time:.2f} seconds")
print(f"CNN - Params: {cnn_params}, Accuracy: {cnn_acc:.4f}, Time: {cnn_time:.2f} seconds")      
        
