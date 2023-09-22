import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#DATASET

# for training:
batch_size = 32



train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)


#####

#1.1
plt.figure()
plt.imshow(train_set[0][0].permute(1,2,0).numpy())
plt.show()

imgs=[]
for i in range(0,100):
    imgs.append(train_set[i][0])

grid_img = torchvision.utils.make_grid(imgs, nrow=10)
print(grid_img.shape)
plt.figure()
plt.imshow(grid_img.permute(1,2,0).numpy())
plt.show()

print(len(train_set))



######### Calculate std and mean
imgs = [item[0] for item in train_set] 
imgs = torch.stack(imgs, dim=0).numpy()

# calculate mean over each channel (r,g,b)
mean_r = imgs[:,0,:,:].mean()
mean_g = imgs[:,1,:,:].mean()
mean_b = imgs[:,2,:,:].mean()
print(mean_r,mean_g,mean_b)

# calculate std over each channel (r,g,b)
std_r = imgs[:,0,:,:].std()
std_g = imgs[:,1,:,:].std()
std_b = imgs[:,2,:,:].std()
print(std_r,std_g,std_b)

transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize((mean_r, mean_g, mean_b), (std_r, std_g, std_b))
    ])

train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, transform=transform, download=True)


#1.3

# Split the training data into training and validation parts.
# here we will use `torch.utils.data.SubsetRandomSampler`.

idx = np.arange(len(train_set))

# Use last 1000 images for validation
val_indices = idx[50000-1000:]
train_indices= idx[:-1000]

print(len(val_indices))
print(len(train_indices))

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          sampler=train_sampler, num_workers=0)

valid_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          sampler=valid_sampler, num_workers=0)


#model

#CLASS
# torch.nn.Conv2d(
# in_channels,
#  out_channels,
#  kernel_size,
#  stride=1,
#  padding=0,
#  dilation=1, 
# groups=1,
#  bias=True,
#  padding_mode='zeros',
#  device=None, dtype=None)

class ConvNet(nn.Module):
    def __init__(self): # just example. super(ConvNet, self).__init__()
        super(ConvNet, self).__init__()
        
        #print()
        #print("in")
        #input = torch.randn(1,3,32,32) #(batch,channels,H,W)
        #print(input.size())

        #print("conv1")
        self.conv1 = nn.Conv2d(3, 32, 3) # 3 channels, 32 filters, 3x3 filter ,input shape (3, 32, 32). 
        #input = self.conv1(input)
        #print(input.size())
        
        #print("conv2")
        self.conv2 = nn.Conv2d(32, 32, 3)
        #input = self.conv2(input)
        #print(input.size())

        #print("pool1")
        self.pool1 = nn.MaxPool2d(2, 2)
        #input = self.pool1(input)
        #print(input.size())

        self.dropout1 = nn.Dropout(0.10)

        #print("conv3")
        self.conv3 = nn.Conv2d(32, 64, 3)
        #input = self.conv3(input)
        #print(input.size())

        #print("conv4")
        self.conv4 = nn.Conv2d(64, 64, 3)
        #input = self.conv4(input)
        #print(input.size())

        #print("pool2")
        self.pool2 = nn.MaxPool2d(2, 2)
        #input = self.pool2(input)
        #print(input.size())

        self.dropout2 = nn.Dropout(0.20)

        #input = input.view(-1,64 * 5 * 5)
        #print("flatten")
        #print(input.size())

        #print("fc1")
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        #input = self.fc1(input)
        #print(input.size())
        self.dropout3 = nn.Dropout(0.50)
        
        #print("fc2")
        self.fc2 = nn.Linear(512, 10) # 10 output classes.
        #input = self.fc2(input)
        #print(input.size())
    
    
    def forward(self, x):
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x)) 
        x = self.pool1(x) # conv, pool.
        x=self.dropout1(x)
        x = F.relu(self.conv3(x)) 
        x = F.relu(self.conv4(x)) 
        x = self.pool2(x) # conv, pool.
        x=self.dropout2(x)
        x = x.view(-1, 64 * 5 * 5) # linearize input "images". 
        x = F.relu(self.fc1(x)) # fully connected.
        x=self.dropout3(x)
        x= self.fc2(x)
        # second layer does not need activation function;
        # softmax is computed by cross entropy loss.
        #x = F.softmax(self.fc2(x),dim=1) # fully connected. 
        #x= nn.Softmax(self.fc2(x))
        return x
    

model= ConvNet()
model = model.to(device)  # put all model params on GPU.

print(model)

learning_rate=0.001
momentum=0.9

# Create loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


# Training

num_epochs = 40
best_val_acc=0
best_epoch=0

train_losses=[]
train_accs=[]
val_losses=[]
val_accs=[]


for epoch in range(1, num_epochs+1):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    run_step = 0
    
    for i, (images, labels) in enumerate(train_loader):
        model.train()
        # shape of input images is (B, 3, 32, 32).
        images = images.to(device)
        labels = labels.to(device)  # shape (B).
        outputs = model(images)  # shape (B, 10).
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()  # reset gradients.
        loss.backward()  # compute gradients.
        optimizer.step()  # update parameters.

        running_loss += loss.item()
        running_total += labels.size(0)

        with torch.no_grad():
            _, predicted = outputs.max(1)
        running_correct += (predicted == labels).sum().item()
        run_step += 1
        if i % 200 == 0:
            # check accuracy.
            print(f'epoch: {epoch}, steps: {i}, '
                  f'train_loss: {running_loss / run_step :.3f}, '
                  f'running_acc: {100 * running_correct / running_total:.1f} %')
            loss_last_value = running_loss/run_step
            acc_last_value = (100 * running_correct / running_total)
            running_loss = 0.0
            running_total = 0
            running_correct = 0
            run_step = 0
    train_losses.append(loss_last_value)
    train_accs.append(acc_last_value)


    # validate
    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad():
        for data in valid_loader:
            
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            run_step += 1
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total

    if(best_val_acc<val_acc):
                best_val_acc=val_acc
                best_epoch=epoch         
    #print(f'running loss: {running_loss:.3f}')
    #print(f'nsteps: {run_step :.3f}') 
    val_losses.append((running_loss/run_step))
    val_accs.append((100 * correct / total))

    print(f'Validation loss: {running_loss/run_step :.3f}')
    print(f'Validation accuracy: {100 * correct / total} %')
    print(f'Validation error rate: {100 - 100 * correct / total: .2f} %')

print('Finished Training')
print(f'Best Validation accuracy: {best_val_acc} %')
print(f'in epoch: {best_epoch} %')



plt.plot(range(1,num_epochs+1), train_losses, c='blue',alpha=0.5, label="train_loss")
plt.plot(range(1,num_epochs+1), val_losses, c='orange',alpha=0.5, label="val_loss")
plt.legend(loc="upper right")
plt.show()



plt.plot(range(1,num_epochs+1), train_accs, c='blue',alpha=0.5,label="train_accuracy")
plt.plot(range(1,num_epochs+1), val_accs, c='orange',alpha=0.5,label="val_accuracy")
plt.legend(loc="upper left")
plt.show()



test_set =  torchvision.datasets.CIFAR10(
    root='./data', train=False, transform = transform)
test_loader = torch.utils.data. DataLoader (dataset=test_set , batch_size =batch_size , shuffle=False)

# TEST
with torch.no_grad():
    correct = 0
    total = 0
    model.eval() 
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device) # shape (B)
        outputs = model(images) # shape (B, num_classes)
        _, predicted = outputs.max(dim=1)
        # predicted.shape: (B)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
    print(f'Test accuracy: {test_acc} %')
    print(f'Test error rate: {100 - 100 * correct / total: .2f} %')