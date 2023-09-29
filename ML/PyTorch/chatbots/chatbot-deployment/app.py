from flask import Flask, render_template, request, jsonify
import json
from flask_cors import CORS
from chat import get_response

app = Flask(__name__)

CORS(app)

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # Check if JSON is valid
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__=="__main__":
    app.run(debug=True)