# Simple-Python-Chatbot

## Creating a simple Python chatbot using natural language processing and deep learning.

This is a chatbot neural network using Keras models that I worked on through a toutorial online by Jere Xu. Given keywords in intents.json it will predict relevent answers to medical-related questions. 

The chatgui file provides an interface to ask questions and recieve responses.

## List of necessary components to run this project

train_chatbot.py — the code for reading in the natural language data into a training set and using a Keras sequential neural network to create a model

chatgui.py — the code for cleaning up the responses based on the predictions from the model and creating a graphical interface for interacting with the chatbot

classes.pkl — a list of different types of classes of responses

words.pkl — a list of different words that could be used for pattern recognition

intents.json — abunch of JavaScript objects that lists different tags that correspond to different types of word patterns

chatbot_model.h5 — the actual model created by train_chatbot.py and used by chatgui.py
