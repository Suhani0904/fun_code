import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
transform=transforms.ToTensor()
train_data=datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_data=datasets.MNIST(root='./data',train=False,download=True,transform=transform)
train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(test_data,batch_size=64,shuffle=False)
model=nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,10)
)
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01)
for epoch in range(5):
    for images,labels in train_loader:
        output=model(images)
        loss=criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
correct=0
total=0
conf_matrix=torch.zeros(10,10)
with torch.no_grad():
    for images,labels in train_loader:
        output=model(images)
        _,pred=torch.max(output,1)
        for t,p in zip(labels,pred):
            conf_matrix[t,p]+=1
        correct+=(pred==labels).sum().item()
        total+=labels.size(0)
print("Test Accuracy:", correct/total)
print("Confusion Matrix:\n", conf_matrix)
