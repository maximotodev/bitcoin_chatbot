import os
import google.generativeai as genai
from dotenv import load_dotenv
# We no longer need render_template, request, session, redirect, url_for for the API part
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS

# --- Configuration & Initialization ---
load_dotenv() # Load environment variables from .env file
API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.0-pro") # Default

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

# --- API Endpoint ---

@app.route('/ask', methods=['POST'])
def ask_bitcoin_api():
    """Handles POST requests with JSON data and returns a JSON response."""
    # Get JSON data from the request body
    data = request.get_json()

    if not data or 'question' not in data:
        return jsonify({"error": "Invalid request. Please provide a 'question' in the JSON body."}), 400 # Bad Request

    user_question = data['question']

    if not user_question:
         return jsonify({"answer": "Please enter a question."}) # Return a friendly message for empty input


    # Combine system prompt and user question
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser Question: {user_question}\n\nAnswer:"

    bot_response = "Sorry, I couldn't process your request." # Default error message

    try:
        # Generate content using the Gemini model
        response = model.generate_content(full_prompt)

        # Process the response
        if response.parts:
             bot_response = response.text.strip()
        else:
             # Handle cases where the response was blocked or incomplete
             block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
             finish_reason = response.candidates[0].finish_reason if response.candidates else "N/A"
             bot_response = f"Response blocked or incomplete. Reason: {block_reason}. Finish: {finish_reason}"
             print(f"Warning: API response incomplete. Feedback: {response.prompt_feedback}, Candidates: {response.candidates}")
             # Consider returning a 500 Internal Server Error if the AI fails unexpectedly
             # return jsonify({"error": bot_response}), 500


    except Exception as e:
        # Catch any API errors and display them
        bot_response = f"An error occurred while contacting the AI: {e}"
        print(f"API Error during generate_content: {e}") # Log the error on the server side
        # Consider returning a 500 Internal Server Error
        # return jsonify({"error": bot_response}), 500


    # Return the response as JSON
    return jsonify({"answer": bot_response, "question": user_question})


# --- Run the Flask App ---
if __name__ == '__main__':
    # No need for os.makedirs('templates', ...) as we are not serving templates
    # Run the development server
    app.run(debug=True) # Runs on http://127.0.0.1:5000/ by default