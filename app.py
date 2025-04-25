import os
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Configuration & Initialization ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.0-pro")

if not API_KEY:
    print("Error: GOOGLE_API_KEY not found.")
    print("Please make sure you have a .env file with GOOGLE_API_KEY=YOUR_API_KEY")
    exit()

try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    exit()

# --- Model Setup ---
try:
    print(f"Attempting to use model: {GEMINI_MODEL_NAME}")
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    # Optional: Test the model configuration
    model.generate_content("test", stream=True)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error creating or testing Gemini model '{GEMINI_MODEL_NAME}': {e}")
    print("Please check your GEMINI_MODEL_NAME in the .env file and ensure it's valid for generateContent.")
    exit()

# --- System Prompt: Define the Chatbot's Persona and Scope ---
# We will prepend this to the history for each turn to maintain persona
SYSTEM_PROMPT = """
You are a specialized Bitcoin chatbot. Your knowledge base is focused ONLY on Bitcoin.
Answer the user's questions accurately and concisely about Bitcoin concepts, history,
technology (blockchain, mining, cryptography), economic aspects (price, market trends, adoption),
notable figures, significant events, and related technical details (like forks, wallets, Layer 2 solutions like Lightning Network).

- Be factual and objective.
- Do NOT give financial advice, investment recommendations, or price predictions.
- If a question is clearly unrelated to Bitcoin (e.g., asking about Ethereum details, other cryptocurrencies unless in direct comparison to Bitcoin, cooking recipes, or general knowledge), politely state that you specialize in Bitcoin and cannot answer the unrelated query.
- Keep answers focused and avoid unnecessary jargon, but explain technical terms if needed.
- Do not engage in speculative discussions about the future price unless citing historical data or widely known analyses (and label them as such).
"""

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Helper to format history for Gemini API ---
def format_history_for_gemini(history):
    """Converts history from frontend format to Gemini API format."""
    gemini_history = []
    # Prepend the system prompt as the initial 'user' turn
    # This helps maintain the persona across turns in start_chat
    gemini_history.append({'role': 'user', 'parts': [{'text': SYSTEM_PROMPT}]})
    # The very first expected response from the model after the system prompt
    # is typically empty or a greeting, which we represent here.
    gemini_history.append({'role': 'model', 'parts': [{'text': 'Okay, I am ready to answer questions about Bitcoin.'}]})


    for message in history:
        role = 'user' if message['type'] == 'user' else 'model'
        gemini_history.append({'role': role, 'parts': [{'text': message['text']}]})
    return gemini_history

# --- API Endpoint ---

@app.route('/ask', methods=['POST'])
def ask_bitcoin_api():
    """Handles POST requests with JSON data and returns a JSON response."""
    data = request.get_json()

    if not data or 'question' not in data or 'history' not in data:
        return jsonify({"error": "Invalid request. Please provide 'question' and 'history' in the JSON body."}), 400 # Bad Request

    user_question = data['question']
    chat_history_frontend = data['history'] # Get history from frontend

    if not user_question:
         return jsonify({"answer": "Please enter a question."}) # Return a friendly message for empty input

    # Format the history for the Gemini API, including the system prompt
    gemini_history = format_history_for_gemini(chat_history_frontend)

    bot_response = "Sorry, I couldn't process your request." # Default error message

    try:
        # Use start_chat with the history
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(user_question)

        # Process the response
        if response.parts:
             bot_response = response.text.strip()
        else:
             # Handle cases where the response was blocked or incomplete
             block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
             finish_reason = response.candidates[0].finish_reason if response.candidates else "N/A"
             bot_response = f"Response blocked or incomplete. Reason: {block_reason}. Finish: {finish_reason}. Please try rephrasing."
             print(f"Warning: API response incomplete. Feedback: {response.prompt_feedback}, Candidates: {response.candidates}")
             # You might want to return a 500 error here in production for genuine failures


    except Exception as e:
        # Catch any API errors and display them
        bot_response = f"An error occurred while contacting the AI: {e}"
        print(f"API Error during send_message: {e}") # Log the error on the server side
        # You might want to return a 500 error here in production

    # Return the response as JSON
    return jsonify({"answer": bot_response}) # Frontend already has the user question

# --- Run the Flask App ---
if __name__ == '__main__':
    app.run(debug=True, port=5000) # Ensure it runs on port 5000