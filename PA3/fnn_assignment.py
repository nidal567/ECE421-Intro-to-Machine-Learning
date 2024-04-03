import torch
#import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

# Function for loading notMNIST Dataset
def loadData(datafile = "/Users/nidaldanial/Downloads/4th Year/2nd Sem/ECE421/PA3/notMNIST.npz"):
    with np.load(datafile) as data:
        Data, Target = data["images"].astype(np.float32), data["labels"]
        np.random.seed(7)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Custom Dataset class.
class notMNIST(Dataset):
    def __init__(self, annotations, images, transform=None, target_transform=None):
        self.img_labels = annotations
        self.imgs = images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

#Define FNN
class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()

        #TODO
        self.fc1 = nn.Linear(784,10)
        self.fc2 = nn.Linear(10,10)
        self.fc3 = nn.Linear(10,10)
        #DEFINE YOUR LAYERS HERE

    def forward(self, x):
        #TODO
        out = torch.flatten(x, start_dim=1) 
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
        #DEFINE YOUR FORWARD FUNCTION HERE

# Commented out IPython magic to ensure Python compatibility.
# Compute accuracy
def get_accuracy(model, dataloader):

    model.eval()
    device = next(model.parameters()).device
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # TODO
            # Return the accuracy
            compare = list()
            output = model(images)
            for outputs in output:
              compare.append(torch.argmax(outputs).tolist())
            compare = torch.tensor(compare)
            compare = compare.to(device)
            temp = torch.eq(labels,compare).type(torch.uint8).sum().tolist()
            total = total + temp 
        accuracy=(total/(len(dataloader)*32))*100

    return accuracy

# Train the model
def train(model, device, learning_rate, train_loader, val_loader, test_loader, num_epochs=50, verbose=False):
  #TODO
  # Define your cross entropy loss function here
  # Use cross entropy loss
  criterion = nn.CrossEntropyLoss()

  #TODO
  # Define your optimizer here
  # Use AdamW optimizer, set the weights, learning rate argument.
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


  acc_hist = {'train':[], 'val':[], 'test': []}

  for epoch in range(num_epochs):
    model = model.train()
    ## training step
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # TODO
        # Follow the step in the tutorial
        ## forward + backprop + loss

        #forward
        outputs= model(images) #to get the images

        ## forward + backprop + loss
        loss= criterion(outputs,labels) #calculate loss
        optimizer.zero_grad()
        loss.backward()


        ## update model params
        optimizer.step()

    model.eval()
    acc_hist['train'].append(get_accuracy(model, train_loader))
    acc_hist['val'].append(get_accuracy(model, val_loader))
    acc_hist['test'].append(get_accuracy(model, test_loader))

    if verbose:
      print('Epoch: %d | Train Accuracy: %.2f | Validation Accuracy: %.2f | Test Accuracy: %.2f')
#%(epoch, acc_hist['train'][-1], acc_hist['val'][-1], acc_hist['test'][-1])

  return model, acc_hist

def experiment(learning_rate=0.0001, num_epochs=50, verbose=False):
  # Use GPU if it is available.
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # Inpute Batch size:
  BATCH_SIZE = 32

  # Convert images to tensor
  transform = transforms.Compose(
      [transforms.ToTensor()])

  # Get train, validation and test data loader.
  trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

  train_data = notMNIST(trainTarget, trainData, transform=transform)
  val_data = notMNIST(validTarget, validData, transform=transform)
  test_data = notMNIST(testTarget, testData, transform=transform)


  train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
  val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

  # Specify which model to use
  model = FNN()

  # Loading model into device
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  model, acc_hist = train(model, device, learning_rate, train_loader, val_loader, test_loader, num_epochs=num_epochs, verbose=verbose)

  # Release the model from the GPU (else the memory wont hold up)
  model.cpu()

  return model, acc_hist

experiment(learning_rate=0.0001, num_epochs=50, verbose=False)

def compare_lr():
  model, acc_hist = experiment(learning_rate=0.1, num_epochs=50, verbose=False)
  plt.figure(figsize=(10,5))
  plt.title("Experiment 1: FNN Training and Test History")
  plt.plot(acc_hist['test'],label="test")
  plt.plot(acc_hist['train'],label="train")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.show()

  model, acc_hist = experiment(learning_rate=0.01, num_epochs=50, verbose=False)
  plt.figure(figsize=(10,5))
  plt.title("Experiment 1: FNN Training and Test History")
  plt.plot(acc_hist['test'],label="test")
  plt.plot(acc_hist['train'],label="train")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.show()

  model, acc_hist = experiment(learning_rate=0.001, num_epochs=50, verbose=False)
  plt.figure(figsize=(10,5))
  plt.title("Experiment 1: FNN Training and Test History")
  plt.plot(acc_hist['test'],label="test")
  plt.plot(acc_hist['train'],label="train")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.show()

  model, acc_hist = experiment(learning_rate=0.0001, num_epochs=50, verbose=False)
  plt.figure(figsize=(10,5))
  plt.title("Experiment 1: FNN Training and Test History")
  plt.plot(acc_hist['test'],label="test")
  plt.plot(acc_hist['train'],label="train")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.show()

compare_lr()