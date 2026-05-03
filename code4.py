import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
transform=transforms.ToTensor()
train_data=datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_data=datasets.MNIST(root='./data',train=False,download=True,transform=transform)
train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
def create_model(init_type="normal"):
    model=nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,128),
        nn.ReLU(),
        nn.Linear(128,10)
    )
    for layer in model:
        if isinstance(layer,nn.Linear):
            if init_type=="normal":
                nn.init.normal_(layer.weight)
            elif init_type=="uniform":
                nn.init.uniform_(layer.weight)
    return model
def train_model(lr,init_type):
    model=create_model(init_type)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr)
    print(f"\nTraining with lr={lr} and init={init_type}")
    for epoch in range(3):
        total_loss=0
        for images,labels in train_loader:
            output=model(images)
            loss=criterion(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
learning_rates=[0.1,0.01,0.001]
inits=["normal","uniform"]
for lr in learning_rates:
    for init in inits:
        train_model(lr,init)       
