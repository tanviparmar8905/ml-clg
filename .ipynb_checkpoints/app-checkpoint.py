from flask import Flask, jsonify
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')   # Use non-GUI backend for server
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import json

app = Flask(__name__)
from flask import send_file
from flask import request

@app.route('/ml/chart/<filename>', methods=['GET'])
def get_chart(filename):
    return send_file(filename, mimetype='image/png')
# Load model artifacts
model = joblib.load("demand_model.pkl")
scaler = joblib.load("demand_scaler.pkl")
label_encoder = joblib.load("demand_label_encoder.pkl")

@app.route('/ml/demand', methods=['GET'])
def test_with_file():
    # Load backend temp JSON file
    with open("response.json") as f:
        data = json.load(f)

    # Extract ML features from nested "mlFeatures"
    ml_features = data["mlFeatures"]

    sample = pd.DataFrame([[ml_features['trend_score'],
                            ml_features['competition_score'],
                            ml_features['price_range']]],
                          columns=['trend_score','competition_score','price_range'])

    # Scale and predict
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)
    pred_label = label_encoder.inverse_transform(prediction)
    proba = model.predict_proba(sample_scaled)
    confidence = np.max(proba)

    # Chart generation (all Module 1 features)
    df = pd.DataFrame([ml_features])  # chart from mlFeatures block
    plt.figure(figsize=(8,5))
    sns.barplot(data=df)
    plt.title("Module 1 Feature Scores")
    chart_path = "module1_chart.png"
    plt.savefig(chart_path)
    plt.close()

    # Merge backend input + ML output + chart reference
    response = data.copy()
    response.update({
        "demand_label": pred_label[0],
        "confidence": round(float(confidence), 2),
        "chart": request.host_url + "ml/chart/" + chart_path

    })

    return jsonify(response)

import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))   # Railway/Render sets PORT automatically
    app.run(host="0.0.0.0", port=port)         # Listen on all interfaces
# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd
# import numpy as np

# app = Flask(__name__)

# # Load model artifacts
# model = joblib.load("demand_model.pkl")
# scaler = joblib.load("demand_scaler.pkl")
# label_encoder = joblib.load("demand_label_encoder.pkl")

# @app.route('/ml/demand', methods=['POST'])
# def predict_demand():
#     data = request.json
    
#     # Convert backend input into DataFrame
#     sample = pd.DataFrame([[data['trend_score'],
#                             data['competition_score'],
#                             data['price_range']]],
#                           columns=['trend_score','competition_score','price_range'])
    
#     # Scale features
#     sample_scaled = scaler.transform(sample)
    
#     # Predict label
#     prediction = model.predict(sample_scaled)
#     pred_label = label_encoder.inverse_transform(prediction)
    
#     # Predict confidence
#     proba = model.predict_proba(sample_scaled)
#     confidence = np.max(proba)
    
#     # Return JSON response
#     return jsonify({
#         "demand_label": pred_label[0],
#         "confidence": round(float(confidence), 2)
#     })

# if __name__ == '__main__':
#     app.run(port=5000, debug=True)