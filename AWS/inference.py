import pandas as pd
import numpy as np
import pickle
from flask import Flask, request
import shap

app = Flask(__name__)
AWS_PORT = 2500

def get_model(filename):
    """Download model from pickle file"""
    return pickle.load(open(filename, 'rb'))


def get_prediction(model, X):
    """Make prediction with the model"""
    return model.predict(X)


# download model
filename = 'churn_model.pkl'
model = get_model(filename)


@app.route('/predict_bankrupt', methods=['POST'])
def predict_churn():
    model = get_model(filename)
    data = request.get_json(force=True)

    df = pd.read_json(data)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    SAMPLE_NUMBER = 0
    score = model.predict_proba(data)[:, 1][SAMPLE_NUMBER]
    score = pd.DataFrame({'feature': 'score', 'importance': [score]})

    ROUNDER = 5
    POSITIVE_CLASS = 1
    feat_name = pd.DataFrame(df.columns)
    feat_values = pd.DataFrame(np.around(shap_values[POSITIVE_CLASS][SAMPLE_NUMBER], ROUNDER))
    df_imp = pd.concat((feat_name, feat_values), axis=1)
    df_imp.columns = ['feature', 'importance']
    df_imp.sort_values(by='importance', inplace=True, ascending=False)

    result = score.append(df_imp, ignore_index = True)

    server_answer = result.to_json()
    return server_answer


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=AWS_PORT)