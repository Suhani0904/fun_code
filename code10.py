import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
transform=transforms.ToTensor()
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,128),
            nn.ReLU(),
            nn.Linear(128,32)
        )
        self.decoder=nn.Sequential(
            nn.Linear(32,128),
            nn.ReLU(),
            nn.Linear(128,784),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x
model=Autoencoder()
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters() ,lr=0.001)
model.train()
losses=[]
for epoch in range(5):
    total_loss=0
    for images,_ in train_loader:
        output=model(images)
        images_flat=images.view(images.size(0),-1)
        loss=criterion(output,images_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    avg_loss=total_loss/len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}, loss:{avg_loss:.4f}")
plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title('Training Loss')
plt.show()     
images,_labels=next(iter(train_loader))
with torch.no_grad():
    output=model(images)
images=images.view(-1,28,28)
output=output.view(-1,28,28)
plt.figure(figsize=(10,4))
for i in range(6): 
    plt.subplot(2,6,i+1)
    plt.imshow(images[i].cpu(),cmap='gray')
    plt.axis('off')

    plt.subplot(2,6,i+7)
    plt.imshow(output[i].cpu(),cmap='gray')
    plt.axis('off')    

