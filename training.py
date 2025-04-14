import random
import json
import pickle
import numpy as np
import warnings
import nltk # natural language toolkit
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

warnings.filterwarnings("ignore")

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('data/intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',', "'"]
# 
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        # Sentence: "How are you?"
        # Tokens: ["How", "are", "you", "?"]
        words.extend(word_list)
        # storing each sentence with its tag i.e "greetings"
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# ** try printing documents for better understanding **

# lemmitizing word i.e running -> run, better -> good
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
# classes = sorted(set(classes))

# storing in files to avoid computation again and again
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))


# data pre-precessing 
training = []
output_empty =  [0] * len(classes)

# converting to numercial values 

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    # insert one to class which belongs to current sentence
    output_row[classes.index(document[1])] = 1
    training.append(bag+output_row)


random.shuffle(training)
training = np.array(training)
train_x = training[:, :len(words)]
train_y = training[:, len(words):]


print("Printing training x")
print(train_x)
print("Printing training y")
print(train_y)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(train_y[0]), activation="softmax")
])

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

history = model.fit(train_x, train_y, epochs=400, batch_size=5, verbose=1)

model.save('chatbot_model.keras', history)
print('done')

