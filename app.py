# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 13:38:58 2022

@author: LENOVO
"""

from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np


app = Flask(__name__)

dic = {0:'buildings',
1:'forest',2:
'glacier',3:
'mountain',4:
'sea',5:
'street'}
model = load_model('model.h5')
model.make_predict_function()

def predict_label(img_path):
    i = mpimg.imread(img_path)
    i = tf.keras.utils.img_to_array(i)/255.0
    i = i.reshape(1,150,150,3)
    p = np.argmax(model.predict(i),axis=1)
    return dic[p[0]]

# routes 
@app.route("/",methods=["GET","POST"])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "this is a data use Case by Moussa DIALLO and SALAH DIA , DIT STUDENTS"

@app.route("/submit",methods=["GET","POST"])
def get_hours():
    if request.method == 'POST':
        img = request.files["my_image"]
        
        img_path = "static/"+ img.filename
        img.save(img_path)
        
        p = predict_label(img_path)
    return render_template("index.html", prediction = p ,img_path = img_path)

if __name__ == '__main__':
    #app.debug = True
    app.run(debug= True)