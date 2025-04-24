from flask import Blueprint, request, jsonify
from chatbot import get_bot_response

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    prompt = data.get("message", "")

    if not prompt:
        return jsonify({"response": "No input provided"}), 400

    response = get_bot_response(prompt)
    return jsonify({"response": response})
