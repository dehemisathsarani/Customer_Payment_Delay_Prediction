from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model, scaler, and feature names
model = pickle.load(open("Model/payment_delay_xgboost_model.pkl", "rb"))
scaler = pickle.load(open("Model/payment_delay_scaler.pkl", "rb"))
feature_names = pickle.load(open("Model/feature_names.pkl", "rb"))

@app.route('/',methods=['POST','GET'])
def index():
    result = None
    if request.method == 'POST':
        invoice_amount = float(request.form['invoice_amount'])
        past_avg_delay_days = float(request.form['past_avg_delay_days'])
        total_invoices = float(request.form['total_invoices'])
        past_late_ratio = float(request.form['past_late_ratio'])

        # Create a feature vector with all required features
        # Initialize with zeros for all features
        features = np.zeros((1, len(feature_names)))
        
        # Set the values for the features we have from the form
        feature_dict = {
            'invoice_amount': invoice_amount,
            'past_avg_delay_days': past_avg_delay_days,
            'total_invoices': total_invoices,
            'past_late_ratio': past_late_ratio
        }
        
        # Populate the feature vector
        for feature_name, value in feature_dict.items():
            if feature_name in feature_names:
                idx = feature_names.index(feature_name)
                features[0, idx] = value
        
        # Apply scaling transformation
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)

        if prediction[0] == 1:
            result = "⚠️ High Risk of Delay"
        else:
            result = "✅ Low Risk"

    return render_template("index.html", prediction_text=result)

if __name__ == '__main__':    
    app.run(debug=True)