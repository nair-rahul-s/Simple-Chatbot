#python -W ignore foo.py
import eel
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
from tensorflow import keras
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import json
import numpy as np
from tensorflow.keras.models import model_from_json
import random

#Loading the saved model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

with open("reqdData.pkl","rb") as pkl:
    dict1 = pickle.load(pkl)
    tokenizer = dict1["tokenizer"]
    classes = dict1["classes"]
    x_test = dict1["test_data"]
    y_test = dict1["test_results"]

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
score = model.evaluate(x_test, y_test, verbose=1)

#loading the JSON data
with open('tsec.json') as json_file:
      data = json.loads(json_file.read())

#CLEANING FUNCTION 1
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

stop_words_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()
my_sw = ['rt', 'ht', 'fb', 'amp', 'gt']


def black_txt_1(token):
	if token == 'u':
		token = 'you'
	return token not in list(string.punctuation) and token not in my_sw #token not in stop_words_ 

def cleaner(word):
	#Remove links
	word = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', 
	            '', word, flags=re.MULTILINE)
	#Decontracted words
	word = decontracted(word)
	#Remove users mentions
	word = re.sub(r'(@[^\s]*)', "", word)
	word = re.sub('[\W]', ' ', word)
	#Lemmatized
	list_word_clean = []
	for w1 in word.split(" "):
		if  black_txt_1(w1.lower()):
			word_lemma =  wn.lemmatize(w1,  pos="v")
			list_word_clean.append(word_lemma)

	#Cleaning, lowering and remove whitespaces
	word = " ".join(list_word_clean)
	word = re.sub('[^a-zA-Z]', ' ', word)
	return word.lower().strip()


########################################################## MODIFIED CODE ###################################################################



ERROR_THRESHOLD = 0.25
context_set = []

def classify(inputText):
    inp = inputText
    if len(cleaner(inp)) != 0:
        inp = cleaner(inp)
    bag = tokenizer.texts_to_sequences([inp])
	#print("BAG : ", bag)
    bag = pad_sequences(bag, maxlen= 10)
    # generate probabilities from the model
    results = model.predict(bag)[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results)] # if r>ERROR_THRESHOLD
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    final_results = []
    final_results.append(results[0])
    final_results.append(results[1])
    final_results.append(results[2])
	#print("RESULTS : ", final_results)
    return_list = []
    for r in final_results:
        return_list.append((classes[r[0]], r[1]))
	#return tuple of intent and probability
    print("Return List : ",return_list)
    return return_list

@eel.expose
def response(inputText):
    context_flag = False
    global context_set
    return_list = classify(inputText)
    optionsList = []
    responsesList = []
    sorryMessage = "Sorry I couldn't understand what you were saying...\nPlease ask a different question"
    #if the context set is empty, check if the top result has context filter and if it has context filter then
    #then dont return contextual response but return error message else append the context_set and return the tag message
    if not context_set:
        for intent in data['intents']:
            if return_list[0][0] == intent['tag']:

                if 'context_filter' in intent:
                    eel.setValue(sorryMessage,"true",optionsList,responsesList)

                else:
                    if 'context_set' in intent and len(intent['context_set'])>0:
                        context_set.append(intent['context_set'])
                    if "options" in intent:
                    	optionsList = intent['options']
                    	responsesList = intent['description']
                    eel.setValue(random.choice(intent['responses']),"true",optionsList,responsesList)
    else:
        for intent in data['intents']:
            for results in return_list:
                if intent['tag'] == results[0] and'context_filter' in intent:
                    if intent['context_filter'] == context_set[0]:
                        context_flag = True
                        context_set.clear()
                        if "options" in intent:
                        	optionsList = intent['options']
                        	responsesList = intent['description']
                        eel.setValue(random.choice(intent['responses']),"true",optionsList,responsesList)
                        break

        if context_flag == False and return_list[0][1] > ERROR_THRESHOLD:
            context_set.clear()
            for intent in data['intents']:
                if intent['tag'] == return_list[0][0] and 'context_filter' in intent:
                    eel.setValue(sorryMessage,"true",optionsList,responsesList)
                    break
                elif intent['tag'] == return_list[0][0]:
                	if "options" in intent:
                		optionsList = intent['options']
                		responsesList = intent['description']
                	eel.setValue(random.choice(intent['responses']),"true",optionsList,responsesList)
                	if 'context_set' in intent and len(intent['context_set'])>0:
                		context_set.append(intent['context_set'])
                	break
                    
        elif context_flag == False and return_list[0][1] < ERROR_THRESHOLD:
            context_set.clear()
            eel.setValue(sorryMessage,"true",optionsList,responsesList)



eel.init('web')
eel.start('ChatUI.html', size=(400,600))


