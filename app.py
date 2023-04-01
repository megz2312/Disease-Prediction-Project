import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence

# app = Flask(__name__)
app = Flask(__name__, template_folder='templates')
# model = pickle.load(open('logreg.pkl', 'rb'))
model = pickle.load(open('kidneymodel.pkl', 'rb'))

@app.route('/')
def home():
    #return render_template('index.html')
    return render_template("kidney.html")

@app.route('/predict',methods=['POST'])
def predict():
    feature_list = request.form.to_dict()
    feature_list = list(feature_list.values())
    feature_list = list(map(float, feature_list))
    final_features = np.array(feature_list).reshape(1, 7) 
#     k=str(final_features)
    prediction = model.predict(final_features)
    predictprob = model.predict_proba(final_features)
    output = int(prediction[0])
    
    if output == 1:
#         text = "Consult doctor immediately. There is a high probability of you getting the disease."
        text="The Best-performing Random Forrest model is {}% confident that the individual has kidney disease.\n\nConsult your doctor immediately.".format(round(predictprob[0][1]*100,2))
    else:
        text="The Best-performing Random Forrest model is {}% confident that the individual does not have kidney disease.\n\nPlease do not be worried.".format(round(predictprob[0][0]*100,2))
        #text = "Do not worry. There is a low probability of you getting the disease."
#     prediction_texts='Model results : '+str(text)
#     prediction_texts="Model Results :\n{}".format(str(text))
    prediction_texts="Model Results :\n\n{}".format(str(text))
#     pred=prediction_texts+'     '+k
#     pred=prediction_text
#     return render_template('index.html', prediction_text='Employee is more likely to {}'.format(text))
#    return render_template('index.html', prediction_text=prediction_texts)
    return render_template('results.html', prediction_text=prediction_texts)



if __name__ == "__main__":
    app.run(debug=True)
