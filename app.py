from flask import Flask, render_template, request, jsonify


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
import json
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from keras.models import load_model
from unidecode import unidecode

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

# Inicializar el lematizador en español
stemmer = SnowballStemmer("spanish")

# Importar los archivos generados en el código anterior
intents = json.loads(open("intents.json", encoding="utf-8").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")


# Pasar las palabras de la oración a su forma raíz
def clean_up_sentence(sentence):
    sentence_words = [stemmer.stem(unidecode(word.lower())) for word in word_tokenize(sentence)]
    return sentence_words


# Convertir la información a unos y ceros según si están presentes en los patrones
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag)


# Predecir la categoría a la que pertenece la oración
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.argmax(res)
    category = classes[max_index]
    return category


# Obtener la respuesta
def get_response(tag, intents_json):
    list_of_intents = intents_json["intents"]
    result = ""
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


def get_Chat_response(text):
    print("Intents: ", intents)
    ints = predict_class(text)
    res = get_response(ints, intents)
    return res

if __name__ == '__main__':
    app.run()