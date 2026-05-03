import torch
import torch.nn as nn
from sklearn.datasets import make_blobs
import torch.optim as optim

X,y=make_blobs(n_samples=100, centers=2, n_features=2, random_state=0)
X=torch.tensor(X, dtype=torch.float32)
model=nn.Linear(2,1)
y=torch.tensor(y,dtype=torch.float32).view(-1,1)
Criterion=nn.BCEWithLogitsLoss()
Optimizer=optim.SGD(model.parameters(), lr=0.01)
for epoch in range(100):
    output=model(X)
    Loss=Criterion(output,y)
    Optimizer.zero_grad()
    Loss.backward()
    Optimizer.step()
print("training complete")   
with torch.no_grad():
    preds = (torch.sigmoid(output) > 0.5).float()
    correct = (preds == y).sum()
    print("Correct predictions:", correct.item())
