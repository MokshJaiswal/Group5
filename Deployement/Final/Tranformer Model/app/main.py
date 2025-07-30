from flask import Flask, request, jsonify
from model_utils import load_model_and_tokenizer, predict_sentiment
from interpret import explain_prediction

app = Flask(__name__)
model, tokenizer, label_encoder = load_model_and_tokenizer()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    prediction = predict_sentiment(text, model, tokenizer, label_encoder)
    explanation = explain_prediction(text, model, tokenizer, label_encoder)
    return jsonify({
        'prediction': prediction,
        'interpretation': explanation
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
