import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
transform=transforms.ToTensor()
train_data=datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_data=datasets.MNIST(root='./data',train=False,download=True,transform=transform)
train_idx=(train_data.targets==0)|(train_data.targets==1)
test_idx=(test_data.targets==0)|(test_data.targets==1)
x_train=train_data.data[train_idx].float()/255.0
y_train=train_data.targets[train_idx].float().view(-1,1)
print(x_train.shape)
print(y_train.shape)
x_test=test_data.data[test_idx].float()/255.0
y_test=test_data.targets[test_idx].float().view(-1,1)
x_train=x_train.view(-1,28*28)
x_test=x_test.view(-1,28*28)
model=nn.Linear(784,1)
criterion=nn.BCEWithLogitsLoss()
optimizer=optim.SGD(model.parameters(),lr=0.1)
for epoch in range(10):
    output=model(x_train)
    loss=criterion(output,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
with torch.no_grad():
    test_preds = (torch.sigmoid(model(x_test)) > 0.5).float()
    test_acc = (test_preds == y_test).sum() / len(y_test)
    print("Test Accuracy:", test_acc.item())
