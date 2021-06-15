# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 20:00:22 2021

@author: Darshan Vithlani
"""

from flask import Flask, render_template, request
import jsonify
import requests
import pandas as pd
import pickle
import numpy as np
import sklearn
from tensorflow import keras
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)
model = keras.models.load_model('my_model.h5')
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

dataframe = pd.read_csv('C:/Users/Darshan Vithlani/Desktop/Backup/Study/Data Science/Mtech/Projects/Bitcoin/BTC-USD.csv', usecols=[4], engine='python')
dataframe = dataframe.iloc[::-1]
dataset = dataframe.values
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

standard_to = StandardScaler()
@app.route("/api", methods=['POST'])
def predict():
    #Fuel_Type_Diesel=0
    result=request.form.to_dict()
    dt=str(result['Year'])
    d0 = date(2021, 4, 10)
    d1 = date(int(dt[:4]), int(dt[5:7]),int(dt[8:]))
    #date(int(dt[6:]), int(dt[3:5]),int(dt[0:2]))
    delta = d1 - d0
    m = delta.days
    if request.method == 'POST':
        #m is the value 
        pred_list=[]
        for i in range(m):
            if i==0:
                gg=float(model.predict(np.array([[[0.95219618]]])))
                pred_list.append(gg)
            else:
                c=list(model.predict(np.reshape(pred_list,(len(pred_list),1,1))))
                for j in range(len(c)):
                    pred_list.append(float(c[j]))
        fort=pred_list[len(pred_list)-1]
        karjat=np.array(fort)
        output=float(scaler.inverse_transform(karjat.reshape(-1,1)))
        return "The Price is "+str(output)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)