# Chatbot-Learning

CHATBOT WITH PYTORCH:

1. Ensure you have chatbot_main, model_chatbot, nltk_utils and train_chatbot in the same folder.
2. nltk_utils contains utility functions and contains the baseline Bag-of-Words model
BoW: https://machinelearningmastery.com/gentle-introduction-bag-words-model/
3. model_chatbot contains the neural network required to train the chatbot 
3a. Load the ‘intents.json’ file
3b. Get the patterns in each intends and tokenize it 
3c. X_train contains the bag of words, Y_train contains the labels 
3d. The neural network will train the bag of words to find the correct labels and produce the loss
4. train_chatbot trains the chatbot and generates a data.pth file, which will be used by chatbot_main as the model
5. Run the programme using chatbot_main
6. New responses can be added in by adding it into the intents.json file – remember to run train_chatbot again to update the data.pth file!

TELEGRAM BOT:

1. From YouTube tutorial https://www.youtube.com/watch?v=NwBWW8cNCP4
