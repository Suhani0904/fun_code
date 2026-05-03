import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
transform = transforms.ToTensor()

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
def add_noise(images):
  noise=torch.randn_like(images)*0.5
  noisy_images=images+noise
  noisy_images=torch.clamp(noisy_images,0.,1.)
  return noisy_images
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
optimizer=optim.Adam(model.parameters(),lr=0.001)
for epoch in range(5):
  total_loss=0
  for images,_ in train_loader:
    noisy_images=add_noise(images)
    output=model(noisy_images)
    images_flat=images.view(images.size(0),-1)
    loss=criterion(output,images_flat)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss+=loss.item()
  print(f"Epoch:{epoch+1} loss:{total_loss/len(train_loader)}")
images,_=next(iter(train_loader))
noisy_images=add_noise(images)
with torch.no_grad():
  output=model(noisy_images)
images=images.view(-1,28,28)
noisy_images=noisy_images.view(-1,28,28)
output=output.view(-1,28,28)

plt.figure(figsize=(10,6))
for i in range(6):
  plt.subplot(3,6,i+1)
  plt.imshow(images[i],cmap='gray')
  plt.title("Clean")
  plt.axis("off")

  plt.subplot(3,6,i+7)
  plt.imshow(noisy_images[i],cmap='gray')
  plt.title("Noisy images")
  plt.axis("off")

  plt.subplot(3,6,i+13)
  plt.imshow(output[i],cmap='gray')
  plt.title("Output")
  plt.axis("off")
plt.show()
