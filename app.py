from flask import Flask, request, jsonify
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
app = Flask(__name__)

# Safe model loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(BASE_DIR, "demand_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "demand_scaler.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "demand_label_encoder.pkl"))
except Exception as e:
    print("Model loading failed:", e)
    model = None
    scaler = None
    label_encoder = None


@app.route('/ml/demand', methods=['GET','POST'])
def predict_demand():
    # Accept POST (live backend JSON) or fallback GET (local response.json)
    if request.method == 'POST':
        data = request.get_json()
    else:
        try:
            with open("response.json") as f:
                data = json.load(f)
        except FileNotFoundError:
            return jsonify({"error": "response.json not found on server"})

    # If models failed to load, return error instead of crashing
    if not model or not scaler or not label_encoder:
        return jsonify({"error": "Model not loaded"})

    # Extract ML features
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

    # Chart generation
    df = pd.DataFrame([ml_features])
    plt.figure(figsize=(8,5))
    sns.barplot(x=list(df.columns), y=list(df.iloc[0]))
    plt.title("Module 1 Feature Scores")
    chart_path = "module1_chart.png"
    plt.savefig(chart_path)
    plt.close()

    # Response back
    response = data.copy()
    response.update({
        "demand_label": pred_label[0],
        "confidence": round(float(confidence), 2),
        "chart_url": request.host_url + "ml/chart/" + chart_path
    })

    return app.response_class(
        response=json.dumps(response, indent=4),
        status=200,
        mimetype="application/json"
    )

@app.route('/ml/chart/<path:filename>')
def serve_chart(filename):
    return app.send_static_file(filename)        # Listen on all interfaces
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