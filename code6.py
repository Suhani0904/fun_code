import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,random_split
transform=transforms.ToTensor()
dataset=datasets.MNIST(root="./data",train=True,transform=transform,download=True)
train_data,val_data=random_split(dataset,[50000,10000])
train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
val_loader=DataLoader(val_data,batch_size=64,shuffle=False)
def small_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,128),
        nn.ReLU(),
        nn.Linear(128,10)
    )
def medium_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,10)
    )
def large_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,256),
        nn.ReLU(),
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,10)
    )
def train_eval(model_fn,name):
    model=model_fn()
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(5):
        for images,labels in train_loader:
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
    model.eval()
    train_correct=0
    train_total=0
    with torch.no_grad():
        for images,labels in train_loader:
            outputs=model(images)
            _,pred=torch.max(outputs.data,1)
            train_total+=labels.size(0)
            train_correct+=(pred==labels).sum().item()
    train_acc=train_correct/train_total
    val_correct=0
    val_total=0
    with torch.no_grad():
        for images,labels in val_loader:
            outputs=model(images)
            _,pred=torch.max(outputs.data,1)
            val_total+=labels.size(0)
            val_correct+=(pred==labels).sum().item()
    val_acc=val_correct/val_total
    print(f"{name}- Train accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}")
    return train_acc,val_acc
small=train_eval(small_model,"Small Model")
medium=train_eval(medium_model,"Medium Model")
large=train_eval(large_model,"Large Model")

        
