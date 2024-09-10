# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Agriculture.pk', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    print(int_features)
   # df1 = pd.DataFrame([int_features]) 
    
    final_features = np.array(int_features)
    
    final_features = final_features.reshape(1,8)
    print(final_features)
    #final_features = pd.DataFrame([final_features])
    print(model.predict(final_features))
    if model.predict(final_features) ==[1]:
       predict = "You Can Proceed Food Relevent Crops"
    else:
       predict = "You can Proceed Horticulture Relevent Crops"
  
    return render_template('index.html',prediction=predict)


if __name__ == "__main__":
    app.run(debug=True)
# -*- coding: utf-8 -*-

