from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialiser l'application Flask
app = Flask(__name__)

# Charger le modèle sauvegardé
model = joblib.load('diabetes_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtenir les données envoyées dans la requête POST (format JSON)
    data = request.get_json()

    # Extraire les variables explicatives
    features = np.array([data['Pregnancies'], data['Glucose'], data['BloodPressure'],
                         data['SkinThickness'], data['Insulin'], data['BMI'],
                         data['DiabetesPedigreeFunction'], data['Age']]).reshape(1, -1)

    # Effectuer la prédiction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].tolist()  # Probabilités des classes

    # Retourner la prédiction sous format JSON
    return jsonify({
        'prediction': int(prediction),  # 0 ou 1 pour indiquer la présence de diabète
        'probability': probability      # Probabilités pour chaque classe
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
