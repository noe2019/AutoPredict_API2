from fastapi import FastAPI
import pickle
import numpy as np

with open("best_model.pkl","rb")  as f:
    model = pickle.load(f)

class_names = np.array(["RIAGENDR","RIDAGEYR","RACE","COUPLE","SMOKER","EDUC","COVERED", "INSURANCE","FAT","Abdobesity","TOTAL_ACCULTURATION_SCORE","HTN"])

app = FastAPI()

@app.get('/')
def reed_root():
    return {'message':'Early diabetes model api'}

@app.post('/predict')
def predict(data:dict):
    """
    predict the class of a given set of features.

    args:
    data (dict): A dictionary containing the features to predict
    e.g. {"features": [1,2,3,4,2,2,3,1,1,1,2]}

    Returns:
     dict: A dictionary containing the predicted class 
    """
    features = np.array(data['features'].reshape(1,-1))
    prediction = model.predict(features)
    class_name = class_names[prediction][0]
    return {'predicted_class':class_name}