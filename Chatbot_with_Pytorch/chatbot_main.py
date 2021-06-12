# Debugging 1: RuntimeError: Tensor for argument #2 'mat1' is on CPU, but expected it to be on GPU (while checking arguments for addmm)
# Debugging 1: https://stackoverflow.com/questions/55983122/pytorch-runtimeerror-expected-object-of-backend-cpu-but-got-backend-cuda-for
"""
test_tensor = torch.from_numpy(test_img)

# Convert to FloatTensor first
test_tensor = test_tensor.type(torch.FloatTensor)

# Then call cuda() on test_tensor
test_tensor = test_tensor.cuda()

log_results = model.forward(test_tensor)
"""

import random
import json
import torch
from model_chatbot import NeuralNet
from nltk_utils import bag_of_words, tokenize

# check for GPU support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)
    
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Eugene's Chatbot"
print("Let's chat! type 'quit'to exit")

while True:
    sentence = input('You: ')
    if sentence == "quit":
        break

    # tokenize the sentence
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    X = X.type(torch.FloatTensor)
    X = X.cuda()

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand")        



    


    





    
    

