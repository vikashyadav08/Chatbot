import json
import numpy as np
import random
import nltk
from django.shortcuts import render
from keras.models import load_model
import pickle

# Load chatbot resources
model = load_model('chat/chatbot_model.h5')  # Adjusted path
intents = json.loads(open('chat/intents.json').read())
words = pickle.load(open('chat/words.pkl', 'rb'))
classes = pickle.load(open('chat/classes.pkl', 'rb'))

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        for i in intents_json['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "I'm not sure I understand. Can you rephrase?"

def chatbot_response(request):
    if request.method == 'POST':
        user_input = request.POST['message']
        ints = predict_class(user_input)
        res = get_response(ints, intents)
        return render(request, 'index.html', {'response': res, 'user_input': user_input})
    return render(request, 'index.html')
