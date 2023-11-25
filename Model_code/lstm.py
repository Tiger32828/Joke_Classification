from collections import deque
from functools import partial
import os, re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold

import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader, Subset, random_split  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For a nice progress bar!
from torch.utils.data import Dataset

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from PorterStemmer import PorterStemmer

class Preprocessor:

    def preprocess(self, filename):
        temp = self.removeSGML(open(filename, encoding="ISO-8859-1").read())
        temp = self.tokenizeText(temp)
        temp = self.removeStopwords(temp)
        return self.stemWords(temp)

    def removeSGML(self, s):
        return re.sub(re.compile(r"<.*?>"), "", s)
        
    def tokenizeText(self, s):
        m = {r"can't":"can not", r"won't":"will not", r"n't":" not", r"'m":" am", r"'s":" is", r"'re":" are", r"'d":" would",
            r"'ll":" will", r"'t":" not", r"'ve":" have"}
        # Create a regular expression from all of the dictionary keys
        regex = re.compile("|".join(map(re.escape, m.keys())))
        # For each match, look up the corresponding value in the dictionary
        new_s = regex.sub(lambda x: m[x.group(0)], s)
        pattern = r"[\d+//+|\d+,+|\d+/.+]+[\d+]+|([A-Z]+\.)+|([A-Za-z]+\-)+[A-Za-z]+|[\w]+"
        return [match.group() for match in re.finditer(pattern, new_s, re.MULTILINE)]

    def removeStopwords(self, wordslist):
        #remove end of line symbol and turn it into hashset
        stopwords = {s[0:len(s)-1] for s in open("stopwords", "r")}
        return [word for word in wordslist if word not in stopwords]
        
    def stemWords(self, wordslist):
        stemmer = PorterStemmer()
        return [stemmer.stem(word, 0, len(word)-1) for word in wordslist]

class CustomizedDataset(Dataset):

    def __init__(self, X, y):
        self.n_samples, self.sequence_length, self.input_size = X.shape

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)
  
    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]

# Recurrent neural network with LSTM (many-to-one)
class RNN_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, sequence_length, num_classes):
        super(RNN_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # because this is a regression task, the output should be one dimension
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out

def hyperparameter_tuning(config, train, verbose=False):
    input_size = train.input_size
    sequence_length = train.sequence_length
    num_layers = 1
    # when num_classes is set to 1, this is essentially a regression task
    num_classes = 6
    hidden_size = config["hidden_size"]
    learning_rate = config["learning_rate"]
    batch_size = int(config["batch_size"])
    num_epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNN_LSTM(input_size, hidden_size, num_layers, sequence_length, num_classes)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    train_data, val_data = random_split(train, [0.8, 0.2])
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
    if verbose: print("Initializing LSTM.")
    # Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
    model = RNN_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)
            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose: print(f"In epoch {epoch:>2} batch {batch_idx:>2}, the training loss is: {round(loss.item(), 10)}")
        val_accuracy = check_accuracy(val_loader, model, verbose=False).item()
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(model.state_dict(), path)
        tune.report(mean_accuracy=val_accuracy)
        if verbose: print(f"After epoch {epoch:>2}, test accuracy is {round(val_accuracy, 10)}")
        

def check_accuracy(loader, model, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            _, true = y.max(1)
            if verbose: print(predictions, true)
            num_correct += (predictions == true).sum()
            num_samples += predictions.size(0)

    # Toggle model back to train
    model.train()
    if verbose: print(num_correct)
    if verbose: print(num_samples)
    return num_correct / num_samples

# Debug mode
debug = True
seed = 486 
n_samples = 2 if debug else 256
torch.manual_seed(seed)

countries = ["American", "Australian", "British", "Chinese", "Russian", "Spanish"]
X, y = [], []
for country in countries:
    for idx, file in enumerate(os.listdir(country)):
        if idx >= n_samples:
            break
        X.append(Preprocessor().preprocess(f"{country}/{file}"))
        y.append(country)

max_length = max([len(x) for x in X])
for i in range(len(X)):
    X[i] = X[i] + [""] * (max_length-len(X[i]))
#X = np.expand_dims(np.array(X), axis=-1)
#y = np.expand_dims(np.array(y), axis=-1)
X = np.array(X).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)
print(X.shape, y.shape)
X = OneHotEncoder().fit_transform(X).toarray().reshape(n_samples*6, max_length, -1)
y = OneHotEncoder().fit_transform(y).toarray().reshape(n_samples*6, -1)
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8)
print(X_train.shape, y_train.shape)

train = CustomizedDataset(X_train, y_train)
#cv = KFold(random_state=seed)
test = CustomizedDataset(X_test, y_test)
'''
config = {
    "batch_size": tune.choice([8, 16, 32]),
    "hidden_size": tune.sample_from(lambda _: 2**np.random.randint(2, 7)),
    "learning_rate": tune.loguniform(1e-4, 1e-1)
}

scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10,
        grace_period=1,
        reduction_factor=2)

result = tune.run(
    partial(hyperparameter_tuning, train=train, verbose=False),
    #metric="accuracy",
    #mode="max",
    resources_per_trial={"cpu": 8, "gpu": 8},
    config=config,
    num_samples=10,
    scheduler=scheduler
)

best_trial = result.get_best_trial(metric="accuracy", mode="max", scope="all")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters
input_size = train.input_size
hidden_size = 16
num_layers = 1
num_classes = 6
sequence_length = train.sequence_length
learning_rate = 0.01
batch_size = 2 if debug else 32
num_epochs = 2 if debug else 40

best_trained_model = RNN_LSTM(input_size, best_trial.config["hidden_size"], num_layers, sequence_length, num_classes)
best_checkpoint_dir = best_trial.checkpoint.value
model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
best_trained_model.load_state_dict(model_state)
test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)
test_acc = check_accuracy(test_loader, best_trained_model)
print("Best trial test set accuracy: {}".format(test_acc))
exit()'''
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters
input_size = train.input_size
hidden_size = 16
num_layers = 1
# when num_classes is set to 1, this is essentially a regression task
num_classes = 6
sequence_length = train.sequence_length
learning_rate = 0.01
batch_size = 2 if debug else 32
num_epochs = 2 if debug else 40

train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)
print("Initializing LSTM.")
# Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
model = RNN_LSTM(input_size, hidden_size, num_layers, sequence_length, num_classes).to(device)

# Loss and optimizer
train_accuracys = []
test_accracys = []
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print("Start training:")
# Train Network
for epoch in range(num_epochs):
    """     for train_idx, val_idx in cv.n_splits(X_train):
        trainset = Subset(train, train_idx)
        valset = Subset(train, val_idx)

        train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False, num_workers=5)
        val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=False, num_workers=5) """

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"In epoch {epoch:>2} batch {batch_idx:>2}, the training loss is: {round(loss.item(), 10)}")
    train_accuracy = check_accuracy(train_loader, model, verbose=False).item()
    test_accuracy = check_accuracy(test_loader, model, verbose=False).item()
    print(f"After epoch {epoch:>2}, the training accuracy is {round(train_accuracy, 10)}, test accuracy is {round(test_accuracy, 10)}")
    train_accuracys.append(train_accuracy)
    test_accracys.append(test_accuracy)
print("Finish training!")
# Check accuracy on training & test to see how good our model
print(train_accuracys)
plt.plot(range(num_epochs), train_accuracys, label="Train accuracy")
plt.plot(range(num_epochs), test_accracys, label="Test accuracy")
plt.legend()
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.show()
plt.savefig("temp.png")
print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")