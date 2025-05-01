import os
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import time

# Import Flask-Limiter and its key function
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address # Helps identify the client by IP

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
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error creating or testing Gemini model '{GEMINI_MODEL_NAME}': {e}")
    print("Please check your GEMINI_MODEL_NAME in the .env file and ensure it's valid for generateContent.")
    exit()

# --- System Prompt: Define the Chatbot's Persona and Scope ---
SYSTEM_PROMPT = """
You are a specialized Bitcoin chatbot. Your knowledge base is focused ONLY on Bitcoin.
Answer the user's questions accurately and concisely about Bitcoin concepts, history,
technology (blockchain, mining, cryptography), economic aspects (market trends, adoption),
notable figures, significant events, and related technical details (like forks, wallets, Layer 2 solutions like Lightning Network).

You are also provided with RECENT Bitcoin data (price, market cap, volume, etc.) at the beginning of the user's input, enclosed by "--- REAL-TIME BITCOIN DATA ---" markers.
**Use this provided Bitcoin data if the user asks questions about the current price, market cap, or trading volume.**
If the provided data indicates a fetch error or is N/A for a specific field, inform the user that real-time data is currently unavailable for that point and provide a general answer based on your training data if possible.
If the user asks about historical prices or trends, you can use your general knowledge but make it clear you are not using the *provided* recent figure.
If the user asks for data *not* included in the provided information (only price, market cap, 24h volume are provided), use your general knowledge if possible or state you only have access to the provided data points.

- Be factual and objective.
- Do NOT give financial advice, investment recommendations, or price predictions beyond interpreting the provided Bitcoin data when asked specifically about *current* values.
- If a question is clearly unrelated to Bitcoin (e.g., asking about Ethereum details, other cryptocurrencies unless in direct comparison to Bitcoin, cooking recipes, or general knowledge), politely state that you specialize in Bitcoin and cannot answer the unrelated query.
- Keep answers focused and avoid unnecessary jargon, but explain technical terms if needed.
"""

# --- Caching Configuration ---
CACHE_DURATION_SECONDS = 300
cached_data = None
last_fetch_time = 0

# --- Helper to fetch real-time data from CoinGecko (uses caching) ---
def get_bitcoin_data():
    global cached_data, last_fetch_time
    current_time = time.time()
    if cached_data and (current_time - last_fetch_time) < CACHE_DURATION_SECONDS:
        print("Using cached data.")
        return cached_data['formatted_string'], None

    print("Fetching fresh data from CoinGecko...")
    api_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_market_cap=true&include_24hr_vol=true&include_last_updated_at=true"

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        if 'bitcoin' in data and 'usd' in data['bitcoin']:
            btc_data = data['bitcoin']
            price = btc_data.get('usd')
            market_cap = btc_data.get('usd_market_cap')
            volume_24h = btc_data.get('usd_24h_vol')
            last_updated_unix = btc_data.get('last_updated_at')

            last_updated_readable = "N/A"
            if last_updated_unix:
                 try:
                    last_updated_readable = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(last_updated_unix))
                 except Exception:
                     pass

            formatted_data_string = "--- RECENT BITCOIN DATA ---\n" # Use RECENT as in prompt
            formatted_data_string += f"Current Price (USD): ${price:,.2f}\n" if price is not None else "Current Price (USD): N/A\n"
            formatted_data_string += f"Market Cap (USD): ${market_cap:,.2f}\n" if market_cap is not None else "Market Cap (USD): N/A\n"
            formatted_data_string += f"24h Volume (USD): ${volume_24h:,.2f}\n" if volume_24h is not None else "24h Volume (USD): N/A\n"
            formatted_data_string += f"Last Updated (UTC): {last_updated_readable}\n"
            formatted_data_string += "--------------------------\n\n"

            cached_data = {
                'formatted_string': formatted_data_string,
                'raw_data': data
            }
            last_fetch_time = current_time

            return formatted_data_string, None

        else:
            fetch_error = "Could not parse Bitcoin data from API."
            print(f"Error: {fetch_error}")
            if cached_data:
                 print("Falling back to stale cached data.")
                 return cached_data['formatted_string'], "Using stale data due to parsing error."
            return "Could not fetch real-time data due to an API response format issue.", fetch_error

    except requests.exceptions.RequestException as e:
        print(f"Error fetching Bitcoin data from CoinGecko: {e}")
        fetch_error = f"Request Error: {e}"
        if cached_data:
            print("Falling back to stale cached data.")
            return cached_data['formatted_string'], "Using stale data due to fetch error."
        return "Could not fetch real-time data due to an API error.", fetch_error

    except Exception as e:
        print(f"An unexpected error occurred fetching Bitcoin data: {e}")
        fetch_error = f"Unexpected Error: {e}"
        if cached_data:
             print("Falling back to stale cached data.")
             return cached_data['formatted_string'], "Using stale data due to unexpected error."
        return "Could not fetch real-time data due to an unexpected error.", fetch_error


# --- Helper to format history for Gemini API ---
def format_history_for_gemini(history):
    gemini_history = []
    gemini_history.append({'role': 'user', 'parts': [{'text': SYSTEM_PROMPT}]})
    gemini_history.append({'role': 'model', 'parts': [{'text': 'Okay, I am ready to answer questions about Bitcoin and use the provided real-time data.'}]})

    for message in history:
        role = 'user' if message['type'] == 'user' else 'model'
        gemini_history.append({'role': role, 'parts': [{'text': message['text']}]})

    return gemini_history

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Limiter Initialization ---
# Configure the limiter. Using 'storage_uri' requires a backend like redis or memcached
# For simple in-memory storage (suitable for single-process deployments like Render's free tier)
# just omit the storage_uri.
# Using a key_func like get_remote_address limits per client IP.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"], # Optional: apply default limits globally
    storage_uri="memory://" # Use in-memory storage
)


# --- API Endpoint ---

@app.route('/ask', methods=['POST'])
# Apply the rate limit specifically to the /ask route
@limiter.limit("10/minute") # This is the core rate limit for the endpoint
def ask_bitcoin_api():
    """Handles POST requests with JSON data and returns a JSON response."""
    data = request.get_json()

    if not data or 'question' not in data or 'history' not in data:
        return jsonify({"error": "Invalid request. Please provide 'question' and 'history' in the JSON body."}), 400

    user_question = data['question']
    chat_history_frontend = data['history']

    if not user_question:
         return jsonify({"answer": "Please enter a question."})

    # --- Fetch Real-time Data (uses cache internally) ---
    realtime_data_string, fetch_error = get_bitcoin_data()

    # --- Prepare Prompt for Gemini ---
    data_block_for_prompt = "--- RECENT BITCOIN DATA ---\n"
    if fetch_error:
         data_block_for_prompt += f"Data Fetch Status: ERROR - {fetch_error}\n"
    else:
         data_block_for_prompt += realtime_data_string.replace("--- RECENT BITCOIN DATA ---\n", "")

    data_block_for_prompt += "--------------------------\n\n"

    prompt_with_data = f"{data_block_for_prompt}User Question: {user_question}\n\nAnswer:"

    # --- Generate Content ---
    bot_response = "Sorry, I couldn't process your request."

    try:
        gemini_history = format_history_for_gemini(chat_history_frontend)
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(prompt_with_data)

        if response.parts:
             bot_response = response.text.strip()
        else:
             block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
             finish_reason = response.candidates[0].finish_reason if response.candidates else "N/A"
             bot_response = f"Response blocked or incomplete. Reason: {block_reason}. Finish: {finish_reason}. Please try rephrasing."
             print(f"Warning: API response incomplete. Feedback: {response.prompt_feedback}, Candidates: {response.candidates}")

    except Exception as e:
        bot_response = f"An error occurred while contacting the AI: {e}"
        print(f"API Error during send_message: {e}")

    # Return the response as JSON
    return jsonify({"answer": bot_response})

# --- Handle Rate Limit Exceeded ---
# Flask-Limiter by default returns a 429 response.
# You can customize the handler if you want a different response body or status code.
# @app.errorhandler(429)
# def ratelimit_handler(e):
#     return jsonify({"error": f"Rate limit exceeded: {e.description}"}), 429


# --- Run the Flask App ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)