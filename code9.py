import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms,models
from torchvision.models import resnet18,ResNet18_Weights
from torch.utils.data import DataLoader
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
train_data=datasets.MNIST(root="./data",train=True,transform=transform,download=True)
test_data=datasets.MNIST(root="./data",train=False,transform=transform,download
=True)
train_loader=DataLoader(train_data,batch_size=64,shuffle=True)  
test_loader=DataLoader(test_data,batch_size=64,shuffle=False)
def transfer_model(num_classes=10):
    model=resnet18(weights=ResNet18_Weights.DEFAULT)
    for params in model.parameters():
        params.requires_grad=False
    model.fc=nn.Linear(model.fc.in_features,num_classes)
    return model
def train_model(model):
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.01)
    model.train()
    for epoch in range(3):
        for images,labels in train_loader:
            images=images.repeat(1,3,1,1)
            outputs=model(images)
            loss=criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
def test_model(model):
    model.evsl()
    correct=0
    total=0
    with torch.no_grad():
        for images,labels in test_loader:
            images=images.repeat(1,3,1,1)
            outputs=model(images)
            _,pred=torch.max(outputs,1)
            correct+=(pred==labels).sum().item()
            total+=labels.size(0)
    return correct/total
model=transfer_model()
train_model(model)
acc=test_model(model)
print(f"Test Accuracy: {acc:.4f}")        
