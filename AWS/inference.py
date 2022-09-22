import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify
import shap

app = Flask(__name__)
AWS_PORT = 8080

# def get_model(filename):
#     """Download model from pickle file"""
#     return pickle.load(open(filename, 'rb'))
#
#
# def get_prediction(model, X):
#     """Make prediction with the model"""
#     return model.predict(X)


# download model
# filename = 'model.pkl'
# model = get_model(filename)

CLF = pickle.load(open('model.pkl', 'rb'))

@app.route('/predict_bankrupt', methods=['GET', 'POST'])
def predict_bankrupt():
    data = request.get_json()
    feats = [' Operating Gross Margin', ' Realized Sales Gross Profit Growth Rate',
       ' Regular Net Profit Growth Rate', ' Gross Profit to Sales',
       ' Cash Reinvestment %', ' Research and development expense rate',
       ' Interest Coverage Ratio (Interest expense to EBIT)',
       ' Equity to Liability', ' Retained Earnings to Total Assets',
       ' Current Ratio', ' Average Collection Days', ' Quick Ratio',
       ' Cash Flow to Sales']

    x = [data[key] for key in feats]
    dict_ = dict(zip(feats, x))

    df = pd.DataFrame([dict_])
    pred = CLF.predict(df)

    return jsonify({'df': str(type(pred))})


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=AWS_PORT)