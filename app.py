#importing libraries
import os
import numpy as np
import flask
from flask import Flask, render_template, request
from sentiment_analysis import max_length, tokenizer_obj
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import pickle


#creating instance of the class
app=Flask(__name__)


@app.route('/')
def welcome():
    return flask.render_template('welcome.html')


#prediction function
def ValuePredictor(to_predict):
    model = pickle.load(open("model.pkl","rb"))
    print("Loaded model from disk")
    result = model.predict(to_predict)
    K.clear_session()
    return result


@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        test_tokens = tokenizer_obj.texts_to_sequences(to_predict_list)
        test_pad = pad_sequences(test_tokens, maxlen = max_length, padding= 'post')
        print(test_pad)
        result = ValuePredictor(test_pad)
        return render_template("result.html",prediction=result)


if __name__ == '__main__':
   app.run(debug= True, port = 5000)
