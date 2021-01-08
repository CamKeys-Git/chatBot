# Simple-Python-Chatbot

## Creating a simple Python chatbot using natural language processing and deep learning.

This is a chatbot neural network using Keras models that I worked on through a toutorial online by Jere Xu. Given keywords in intents.json it will predict relevent answers to medical-related questions. 

## List of necessary components to run this project

#### train_chatbot.py — the code for reading in the natural language data into a training set and using a Keras neural network to create a model

#### chatgui.py — the code for cleaning up the responses based on the predictions from the model and creating a graphical interface for interacting with the chatbot

#### classes.pkl — a list of different types of classes of responses

#### words.pkl — a list of different words that could be used for pattern recognition

#### intents.json — a bunch of JavaScript objects that lists different tags that correspond to different types of word patterns

#### chatbot_model.h5 — the actual model created by train_chatbot.py and used by chatgui.py


Initialize all of the lists where natural language data is stored.

```python
for intent in intents['intents']:
    for pattern in intent['patterns']:

        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
```            
            
