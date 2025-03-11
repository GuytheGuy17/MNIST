# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 17:55:47 2025

@author: gmnmc
"""

import time
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

#%%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), std=(0.3081,))  
])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

#%%
image, label = train_data[0]
print(image.size())
#%%
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate model, loss function, and optimizer
model = MNISTCNN()
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

num_epochs = 10  
all_losses = []  

for epoch in range(num_epochs):
    epoch_loss = 0.0
    start_time = time.time()  
    
 
    for images, labels in train_loader:
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        
        optimizer.zero_grad()  
        loss.backward()       
        optimizer.step()       
        
        # Accumulate loss over this epoch
        epoch_loss += loss.item()
    
    # Average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    all_losses.append(avg_epoch_loss)
    
    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_epoch_loss:.4f} - Time: {elapsed_time:.2f}s")
   
    
#%%
torch.save(model.state_dict(), 'trained_net_1.pth')
#%%
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100*correct/total

print(f"Model accuracy: {accuracy:.2f}%")