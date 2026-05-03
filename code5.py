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
def create_model():
    model=nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28,128),
        nn.ReLU(),
        nn.Linear(128,10)
    )
    return model
def train_model(optimizer_name):
    model=create_model()
    criterion=nn.CrossEntropyLoss()
    if optimizer_name=="SGD":
        optimizer=optim.SGD(model.parameters(),lr=0.01)
    elif optimizer_name=="Momentum":
        optimizer.SGD(model.parameters(),lr=0.01,momentum=0.9)
    elif optimizer_name=="Adam":
        optimizer=optim.Adam(model.parameters(),lr=0.001)
    elif optimizer_name=="RMSprop":
        optimizer=optim.RMSprop(model.parameters(),lr=0.001)
    train_losses=[]
    test_losses=[]
    for epoch in range(5):
        total_train_loss=0
        for images,labels in train_loader:
            output=model(images)
            loss=criterion(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss+=loss.item()
        train_losses.append(total_train_loss)
        total_test_loss=0
        correct=0
        total=0
        with torch.no_grad():
            for images,labels in test_loader:
                outputs=model(images)
                loss=criterion(outputs,labels)
                total_test_loss+=loss.item()    
                _,pred=torch.max(output,1)
                correct+=(pred==labels).sum().item()
                total+=labels.size(0)
        test_losses.append(total_test_loss)        
    return train_losses,test_losses
optimizers=["SGD","Momentum","Adam","RMSprop"]
results={}
for opt in optimizers:
    train_loss,test_loss=train_model(opt)
    results[opt]={train_1,test_1}
print(results)    
