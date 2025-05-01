import os
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import time
import logging # Import the logging module

# Import Flask-Limiter if you added it in a previous branch
# from flask_limiter import Limiter
# from flask_limiter.util import get_remote_address

# --- Configure Logging ---
# Get the root logger
logger = logging.getLogger()
# Set the minimum level to log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.INFO) # Use INFO for general operation, DEBUG for detailed troubleshooting

# Create a handler to output logs to the console (stderr by default)
handler = logging.StreamHandler()
# Define the log message format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger (prevent duplicate handlers if hot-reloading)
if not logger.handlers:
    logger.addHandler(handler)


# --- Configuration & Initialization ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.0-pro")

if not API_KEY:
    # Use logging.critical for errors that prevent the app from starting
    logger.critical("Error: GOOGLE_API_KEY not found.")
    logger.critical("Please make sure you have a .env file with GOOGLE_API_KEY=YOUR_API_KEY")
    exit()

try:
    genai.configure(api_key=API_KEY)
    logger.info("Gemini API configured successfully.")
except Exception as e:
    # Use logging.exception to log the error including the traceback
    logger.exception("Error configuring Gemini API:")
    exit()

# --- Model Setup ---
model = None # Initialize model as None
try:
    logger.info(f"Attempting to use model: {GEMINI_MODEL_NAME}")
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    # Optional: Test the model configuration
    # try:
    #     model.generate_content("test", stream=True)
    #     logger.info("Model test successful.")
    # except Exception as test_e:
    #      logger.warning(f"Model test failed: {test_e}") # Log test failure as warning
    logger.info("Gemini model loaded successfully.")
except Exception as e:
    logger.exception(f"Error creating or testing Gemini model '{GEMINI_MODEL_NAME}':")
    logger.critical("Please check your GEMINI_MODEL_NAME in the .env file and ensure it's valid for generateContent.")
    # It's better to let the app start but return an error on /ask if model loading fails
    # exit() # Don't exit here, let the /ask route handle the missing model


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
        logger.debug("Using cached data for CoinGecko.") # Use debug for frequent events
        return cached_data['formatted_string'], None

    logger.info("Fetching fresh data from CoinGecko...")
    api_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_market_cap=true&include_24hr_vol=true&include_last_updated_at=true"

    try:
        response = requests.get(api_url, timeout=10) # Add a timeout
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
                     logger.warning("Failed to format last updated timestamp.")
                     pass

            formatted_data_string = "--- RECENT BITCOIN DATA ---\n"
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
            logger.info("Successfully fetched and cached new CoinGecko data.")
            return formatted_data_string, None

        else:
            fetch_error = "Could not parse Bitcoin data from CoinGecko API response."
            logger.error(fetch_error)
            if cached_data:
                 logger.warning("Falling back to stale cached data after parsing error.")
                 return cached_data['formatted_string'], "Using stale data due to parsing error."
            return "Could not fetch real-time data due to an API response format issue.", fetch_error

    except requests.exceptions.RequestException as e:
        logger.exception("Error fetching Bitcoin data from CoinGecko:") # Log the full exception
        fetch_error = f"Request Error: {e}"
        if cached_data:
            logger.warning("Falling back to stale cached data after fetch error.")
            return cached_data['formatted_string'], "Using stale data due to fetch error."
        return "Could not fetch real-time data due to an API error.", fetch_error

    except Exception as e:
        logger.exception("An unexpected error occurred fetching Bitcoin data:") # Log the full exception
        fetch_error = f"Unexpected Error: {e}"
        if cached_data:
             logger.warning("Falling back to stale cached data after unexpected error.")
             return cached_data['formatted_string'], "Using stale data due to unexpected error."
        return "Could not fetch real-time data due to an unexpected error.", fetch_error


# --- Helper to format history for Gemini API ---
def format_history_for_gemini(history):
    gemini_history = []
    # Prepend the system prompt as the initial 'user' turn
    gemini_history.append({'role': 'user', 'parts': [{'text': SYSTEM_PROMPT}]})
    gemini_history.append({'role': 'model', 'parts': [{'text': 'Okay, I am ready to answer questions about Bitcoin and use the provided real-time data.'}]})

    # Add previous chat turns
    for message in history:
        role = 'user' if message['type'] == 'user' else 'model'
        gemini_history.append({'role': role, 'parts': [{'text': message['text']}]})

    return gemini_history

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Limiter Initialization (if you added it) ---
# If you added the rate limiting feature in a previous step, uncomment this block:
# limiter = Limiter(
#     get_remote_address,
#     app=app,
#     default_limits=["200 per day", "50 per hour"],
#     storage_uri="memory://"
# )

# --- API Endpoint ---

@app.route('/ask', methods=['POST'])
# Apply the rate limit decorator if you added the feature
# @limiter.limit("10/minute")
def ask_bitcoin_api():
    """Handles POST requests with JSON data and returns a JSON response."""
    # Check if model was initialized successfully on startup
    if model is None:
         logger.error("Gemini model not initialized.")
         return jsonify({"answer": "The chatbot backend is not fully initialized. Please try again later or contact the administrator."}), 503 # Service Unavailable

    logger.info("Received /ask request.")
    try:
        data = request.get_json()

        if not data or 'question' not in data or 'history' not in data:
            logger.warning("Invalid request: Missing 'question' or 'history' in JSON body.")
            return jsonify({"error": "Invalid request. Please provide 'question' and 'history' in the JSON body."}), 400 # Bad Request

        user_question = data['question']
        chat_history_frontend = data['history']

        if not user_question:
             logger.info("Received empty user question.")
             return jsonify({"answer": "Please enter a question."})

        logger.info(f"User Question: {user_question}")
        logger.debug(f"Chat History received ({len(chat_history_frontend)} turns): {chat_history_frontend}") # Log history for debugging

        # --- Fetch Real-time Data (uses cache internally) ---
        realtime_data_string, fetch_error = get_bitcoin_data()

        # --- Prepare Prompt for Gemini ---
        data_block_for_prompt = "--- RECENT BITCOIN DATA ---\n"
        if fetch_error:
             data_block_for_prompt += f"Data Fetch Status: ERROR - {fetch_error}\n"
             logger.warning(f"Data fetch error included in prompt: {fetch_error}")
        else:
             data_block_for_prompt += realtime_data_string.replace("--- RECENT BITCOIN DATA ---\n", "")

        data_block_for_prompt += "--------------------------\n\n"

        prompt_with_data = f"{data_block_for_prompt}User Question: {user_question}\n\nAnswer:"
        logger.debug(f"Full prompt sent to Gemini:\n---\n{prompt_with_data}\n---")

        # --- Generate Content ---
        bot_response = "Sorry, I couldn't process your request."

        try:
            gemini_history = format_history_for_gemini(chat_history_frontend)
            chat = model.start_chat(history=gemini_history)

            # Send the *current* user question along with the real-time data string
            response = chat.send_message(prompt_with_data)
            logger.info("Gemini send_message call successful.")

            if response.parts:
                 bot_response = response.text.strip()
                 logger.info("Successfully generated bot response.")
                 logger.debug(f"Bot Response: {bot_response}")
            else:
                 block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                 finish_reason = response.candidates[0].finish_reason if response.candidates else "N/A"
                 bot_response = f"Response blocked or incomplete. Reason: {block_reason}. Finish: {finish_reason}. Please try rephrasing."
                 logger.warning(f"API response incomplete or blocked. Prompt Feedback: {response.prompt_feedback}, Candidates: {response.candidates}")
                 # Although we return a user message, consider a 500 for critical AI failures
                 # return jsonify({"answer": bot_response}), 500

        except Exception as e:
            logger.exception("An error occurred during Gemini send_message:") # Log the full exception
            bot_response = f"An internal error occurred while contacting the AI: {e}" # More generic for user
            # Consider returning 500 Internal Server Error for unhandled exceptions
            # return jsonify({"answer": bot_response}), 500

        # Return the response as JSON
        return jsonify({"answer": bot_response})

    except Exception as e:
        # Catch any unexpected errors within the route handling itself
        logger.exception("An unexpected error occurred in the /ask route handler:")
        # Return 500 Internal Server Error for errors not specifically handled above
        return jsonify({"error": "An unexpected internal server error occurred."}), 500


# --- Handle Rate Limit Exceeded (if you added it) ---
# If you added the rate limiting feature, uncomment and potentially customize this block:
# @app.errorhandler(429)
# def ratelimit_handler(e):
#     logger.warning(f"Rate limit exceeded for client {get_remote_address()}: {e.description}")
#     return jsonify({"error": f"Too many requests. Please try again later."}), 429 # More user-friendly message


# --- Handle Generic Internal Server Error ---
@app.errorhandler(500)
def internal_error(error):
    # This catches unhandled exceptions that result in a 500 error
    logger.exception("Unhandled internal server error caught by error handler:")
    return jsonify({"error": "An unexpected internal server error occurred."}), 500


# --- Run the Flask App ---
if __name__ == '__main__':
    # Ensure the venv is activated before running
    # In production (like Render), a WSGI server like Gunicorn runs the app,
    # which typically handles logging differently or relies on stderr/stdout.
    # The logging configuration above will work with Gunicorn as it writes to stderr.
    app.run(debug=True, port=5000)