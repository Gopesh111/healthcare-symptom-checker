# app.py — Flask backend
from flask import Flask, request, jsonify
from llm_wrapper import get_safely_inferred
import json, os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "Healthcare Symptom Checker — POST /api/symptom-check with {'symptoms':'...'}"

@app.route("/api/symptom-check", methods=["POST"])
def symptom_check():
    data = request.get_json(force=True, silent=True)
    if not data or "symptoms" not in data:
        return jsonify({"error": "Please POST JSON with 'symptoms' field."}), 400

    symptoms = data["symptoms"]
    result = get_safely_inferred(symptoms, allow_llm=True)
    return jsonify(result.dict())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
