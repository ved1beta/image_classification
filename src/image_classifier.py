import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


num_epochs =100
batch_size = 32
learning_rate = 0.001

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)


device = "cuda" if torch.cuda.is_available() else "cpu"

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 =nn.Linear(64*4*4, 64)
        self.fc2 = nn.Linear(64,10)
    
    def forward(self, x):
        # N, 3, 32, 32
        x = F.relu(self.conv1(x))   # -> N, 32, 30, 30
        x = self.pool(x)            # -> N, 32, 15, 15
        x = F.relu(self.conv2(x))   # -> N, 64, 13, 13
        x = self.pool(x)            # -> N, 64, 6, 6
        x = F.relu(self.conv3(x))   # -> N, 64, 4, 4
        x = torch.flatten(x, 1)     # -> N, 1024
        x = F.relu(self.fc1(x))     # -> N, 64
        x = self.fc2(x)             # -> N, 10
        return x
    
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


train_losses = []
accuracies = []

def imshow(img):
    img = img / 2 + 0.5  
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Step [{i + 1}/{n_total_steps}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Accuracy: {100 * correct / total:.2f}%')
    

    epoch_loss = running_loss / n_total_steps
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    accuracies.append(epoch_acc)
    

    if epoch % 2 == 0:  # Show every 2 epochs
        model.eval()
        with torch.no_grad():
            
            dataiter = iter(test_loader)
            images, labels = next(dataiter)
            images = images.to(device)
            
           
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            imshow(make_grid(images[:8]))
            plt.title('Sample Predictions\nActual: ' + ' '.join(f'{classes[labels[j]]}' for j in range(8)) + 
                     '\nPredicted: ' + ' '.join(f'{classes[predicted[j]]}' for j in range(8)))
            
          
            plt.subplot(1, 2, 2)
            plt.plot(train_losses, label='Training Loss')
            plt.plot(accuracies, label='Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend()
            plt.title('Training Progress')
            plt.tight_layout()
            plt.show()

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy over Time')
plt.legend()
plt.tight_layout()
plt.show()    