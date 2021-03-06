# Simple-Python-Chatbot

## A simple Python chatbot using natural language processing and deep learning.

This is a 3-layer neural network using Keras models that I worked on through a toutorial online by Jere Xu. Given keywords in intents.json it will create a training model and learn to predict relevent answers to medical-related questions. It also uses tkinter to create a GUI and extract the information from the files.

## List of necessary components to run this project

train_chatbot.py — the code for reading in the natural language data into a training set and using a Keras neural network to create a model

chatgui.py — the code for cleaning up the responses based on the predictions from the model and creating a graphical interface for interacting with the chatbot

classes.pkl — a list of different types of classes of responses

words.pkl — a list of different words that could be used for pattern recognition

intents.json — a bunch of JavaScript objects that lists different tags that correspond to different types of word patterns

chatbot_model.h5 — the actual model created by train_chatbot.py and used by chatgui.py

## Run The Program

To run the chatbot first train the model with train_chatbot.py then run chatgui.py.

```terminal
python train_chatbot.py
python chatgui.py
```

Once you run the program you should get a pop-up GUI to comunicate with the trained chatbot.

## How It Works


The code below will initialize all of the lists in intents.json where the natural language data is stored.

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
            
This will create model with 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons equal to number of intents to predict output intent with softmax.

```python
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
```

Training data is initialized with the variable training. Then the words are lemmatized using lemmatizer.lemmatize and added to the bag list. 

```python
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
```    
    
 A dataset is created for training.   
    
```python
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")
```

