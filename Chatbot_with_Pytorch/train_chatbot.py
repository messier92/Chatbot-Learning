# debugging 1: RuntimeError: DataLoader worker (pid(s) 18916, 20180) exited unexpectedly
# debugging 1: https://github.com/pytorch/pytorch/issues/5301
# debugging 1: dataset = UP_Dataset() // train_loader = DataLoader(dataset=dataset, batch_size=100, shuffle=True, num_workers=0) # change num_workers=0

# debugging 2: RuntimeError: Expected object of scalar type Long but got scalar type Float when using CrossEntropyLoss
# debugging 2: https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-when-using-crossentropyloss/30542/9
# debugging 2: loss = criterion(output, target.long())

import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_chatbot import NeuralNet

all_words = []
tags = []
xy = []

# load the json file
with open('intents.json','r') as f:
    intents = json.load(f)

# for each intent that is tagged 'intents'
for intent in intents['intents']:
    # get the tags for each intent
    tag = intent['tag']
    tags.append(tag)
    # get the patterns in each intent and tokenize it
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))

ignore_words = ['?','!','.',',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

#print(all_words)
#print(tags)
#print(intents)

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) #CrossEntropyLoss

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000
#print(input_size, len(all_words))
#print(output_size, tags)


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        #forward
        outputs = model(words)
        loss = criterion(outputs, labels.long())

        #backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # every 100 steps
    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')


print(f'final loss, loss={loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

#save to a pickled file
FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file is saved to {FILE}')
