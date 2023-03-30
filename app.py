import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence

app = Flask(__name__)
# model = pickle.load(open('logreg.pkl', 'rb'))
model = pickle.load(open('kidneymodel.pkl', 'rb'))
cols=['tenure', 'SeniorCitizen', 'Partner','gender','PhoneService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'PaperlessBilling', 'InternetService_DSL', 'InternetService_Fiber optic']
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    feature_list = request.form.to_dict()
    feature_list = list(feature_list.values())
    feature_list = list(map(float, feature_list))
    final_features = np.array(feature_list).reshape(1, 7) 
#     k=str(final_features)
    prediction = model.predict(final_features)
    output = int(prediction[0])
    
    if output == 1:
        text = "Consult doctor immediately. There is a high probability of you getting the disease."
    else:
        text = "Do not worry. There is a low probability of you getting the disease."
    
    prediction_texts='Model results : '+str(text)
#     pred=prediction_texts+'     '+k
#     pred=prediction_text
#     return render_template('index.html', prediction_text='Employee is more likely to {}'.format(text))
    return render_template('index.html', prediction_text=prediction_texts)



if __name__ == "__main__":
    app.run(debug=True)
