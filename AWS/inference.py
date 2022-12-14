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



@app.route('/predict_bankrupt', methods=['GET', 'POST'])
def predict_bankrupt():
    model = pickle.load(open('model.pkl', 'rb'))
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
    score = model.predict_proba([x])[0][1]
    # score_df = pd.DataFrame({'feature': 'score', 'importance': [score]})
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    ROUNDER = 5
    feat_name = pd.DataFrame(df.columns)
    feat_values = pd.DataFrame(np.around(shap_values[0], ROUNDER)).T
    df_imp = pd.concat((feat_name, feat_values), axis=1)
    df_imp.columns = ['feature', 'importance']
    df_imp.sort_values(by='importance', inplace=True, ascending=False)

    feat_import = df_imp.set_index('feature')['importance'].to_dict()
    score = {'score': 1 - score}

    return jsonify(feat_import, score)


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=AWS_PORT)