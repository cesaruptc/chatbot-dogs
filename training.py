import random
import json
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from unidecode import unidecode

# Descargar recursos necesarios para NLTK (si no los has descargado)
nltk.download("stopwords")
nltk.download("punkt")

# Inicializar el lematizador
stemmer = SnowballStemmer("spanish")

# Cargar los datos de intents.json
intents = json.loads(open("intents.json").read())

# Procesamiento de datos
words = []
classes = []
documents = []
ignore_letters = ["?", "!", "¿", ".", ",", "á", "é", "í", "ó", "ú", "ü", "ñ"]

# Obtener palabras vacías (stopwords) en español
stop_words = set(stopwords.words("spanish"))

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Tokenización y lematización en español
        word_list = [
            stemmer.stem(unidecode(word.lower()))
            for word in word_tokenize(pattern)
            if word not in ignore_letters and word not in stop_words
        ]
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = sorted(set(words))

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Creación de datos de entrenamiento
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = [0] * len(words)
    word_patterns = [stemmer.stem(unidecode(word.lower())) for word in document[0]]
    for word in words:
        bag[words.index(word)] = 1 if word in word_patterns else 0

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

np.random.seed(42)

random.shuffle(training)

# Convierte las listas en arrays NumPy
train_x = np.array([entry[0] for entry in training])
train_y = np.array([entry[1] for entry in training])


# Definición del modelo de la red neuronal
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))  # capa densa
model.add(Dropout(0.5))  # drop out
model.add(Dense(64, activation="relu"))  # Densa
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))  # Densa de salida

# Compilación del modelo
sgd = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Entrenamiento del modelo
model.fit(train_x, train_y, epochs=100, batch_size=2, verbose=1)

# Guarda el modelo entrenado
model.save("chatbot_model.h5")
