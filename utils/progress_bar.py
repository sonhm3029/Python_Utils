from tqdm import trange, tqdm
import time
import functools
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam

# Simple
epochs = 10
with trange(epochs) as pbar:
    for epoch in pbar:
        time.sleep(0.5)
        pbar.set_description(f"Epoch [{epoch}/{epochs}]")
        
    
class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081, ))
])
dataset = datasets.MNIST('./mnist_data', train=True, download=True,transform=transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Perceptron().to(device)

optimizer = Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
batch_size = 64
epochs = 10

train_loader = DataLoader(dataset,
                          batch_size=batch_size,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

model.train()
for epoch in range(epochs):
    with tqdm(train_loader, unit='batch') as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
        
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            loss = F.nll_loss(output, target)
            correct = (predictions == target).sum().item()
            accuracy = correct/batch_size
            
            loss.backward()
            optimizer.step()
            
            tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
            time.sleep(0.1)