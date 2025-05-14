import os
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import time
import logging
import re # Import regex for potentially more robust keyword matching

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
    logger.info("Gemini model loaded successfully.")
except Exception as e:
    logger.exception(f"Error creating or testing Gemini model '{GEMINI_MODEL_NAME}':")
    logger.critical("Please check your GEMINI_MODEL_NAME in the .env file and ensure it's valid for generateContent.")


# --- System Prompt: Define the Chatbot's Persona and Scope ---
# *** MODIFIED PROMPT STARTS HERE ***
SYSTEM_PROMPT = """
CRITICAL INSTRUCTION: You are a specialized Bitcoin chatbot. Your knowledge base is focused STRICTLY and ONLY on Bitcoin concepts, history, technology, economics, figures, events, and technical details (blockchain, mining, cryptography, wallets, Layer 2 like Lightning Network).

Your role is purely informative and factual. UNDER NO CIRCUMSTANCES provide financial advice, investment recommendations, or price predictions. Do not speculate on future price movements or recommend buying/selling decisions.

You are provided with RECENT Bitcoin data under "--- RECENT BITCOIN DATA ---".
- When responding to questions about CURRENT price, market cap, or 24h volume, ALWAYS use the provided data EXACTLY.
- Begin answers about current metrics by referencing the data, e.g., "According to the recent data provided, the current price is...".
- If the data block indicates an ERROR or a value is N/A, state "I could not fetch real-time data for that at this moment" before giving any general information based on your training. Do NOT use outdated general knowledge for CURRENT figures.

Answer user questions accurately and concisely. Aim for clarity and avoid unnecessary jargon; explain technical terms (like hashrate, UTXO, difficulty) when first used or requested. Use Markdown formatting (bold, lists) for readability when appropriate.

If a question is:
1. Clearly outside Bitcoin scope (other crypto details, general knowledge): Politely state: "I am a specialized chatbot focused only on Bitcoin, and I cannot answer questions about [topic]."
2. About comparing Bitcoin to another crypto feature: Briefly explain the Bitcoin aspect and mention the comparison point without detailing the other crypto.
3. Ambiguous or you lack sufficient information: State that you do not have enough information to provide a confident answer on that specific topic, or politely ask for clarification if appropriate.

Maintain a helpful, informative, and neutral tone. Do not express opinions or engage in speculative discussions.
"""
# *** MODIFIED PROMPT ENDS HERE ***


# --- Caching Configuration (Keep it) ---
CACHE_DURATION_SECONDS = 300
cached_data = None
last_fetch_time = 0

# --- Helper to fetch real-time data (Keep it) ---
def get_bitcoin_data():
    global cached_data, last_fetch_time
    current_time = time.time()
    if cached_data and (current_time - last_fetch_time) < CACHE_DURATION_SECONDS:
        logger.debug("Using cached data for CoinGecko.")
        return cached_data['formatted_string'], None

    logger.info("Fetching fresh data from CoinGecko...")
    api_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_market_cap=true&include_24hr_vol=true&include_last_updated_at=true"

    try:
        response = requests.get(api_url, timeout=10)
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
        logger.exception("Error fetching Bitcoin data from CoinGecko:")
        fetch_error = f"Request Error: {e}"
        if cached_data:
            logger.warning("Falling back to stale cached data after fetch error.")
            return cached_data['formatted_string'], "Using stale data due to fetch error."
        return "Could not fetch real-time data due to an API error.", fetch_error

    except Exception as e:
        logger.exception("An unexpected error occurred fetching Bitcoin data:")
        fetch_error = f"Unexpected Error: {e}"
        if cached_data:
             logger.warning("Falling back to stale cached data after unexpected error.")
             return cached_data['formatted_string'], "Using stale data due to unexpected error."
        return "Could not fetch real-time data due to an unexpected error.", fetch_error


# --- Helper to format history for Gemini API (Keep it) ---
def format_history_for_gemini(history):
    gemini_history = []
    # The first turn should be the system prompt
    gemini_history.append({'role': 'user', 'parts': [{'text': SYSTEM_PROMPT}]})
    # The AI's expected initial response after the system prompt
    gemini_history.append({'role': 'model', 'parts': [{'text': 'Okay, I am ready to answer questions about Bitcoin and use the provided real-time data.'}]}) # This should generally align with the prompt's capabilities

    # Add previous chat turns
    for message in history:
        role = 'user' if message['type'] == 'user' else 'model'
        gemini_history.append({'role': role, 'parts': [{'text': message['text']}]})

    return gemini_history

# --- Define Introductory Tour Content (Keep it) ---
INTRO_TOUR_MESSAGES = [
    "Welcome to the Bitcoin Introduction Tour!",
    "Bitcoin is a decentralized digital currency, meaning it operates without a central bank or single administrator.",
    "It was invented by an unknown person or group using the name Satoshi Nakamoto and released in January 2009.",
    "The core technology behind Bitcoin is called the **blockchain**. Think of it as a public digital ledger that records all Bitcoin transactions across a network of computers.",
    "These transactions are verified by network participants through a process called **mining**. Miners use computing power to solve complex puzzles, and the first one to solve it gets to add the next block of transactions to the blockchain and earn newly created Bitcoin (plus transaction fees).",
    "Bitcoin's supply is limited to **21 million coins**, which makes it a scarce asset. This scarcity is a key part of its economic model.",
    "You can send and receive Bitcoin using a **digital wallet**. There are different types of wallets, from software apps on your phone/computer to physical hardware devices.",
    "Bitcoin is volatile, meaning its price can change significantly and rapidly. It's considered a high-risk asset.",
    "Important Disclaimer: I cannot provide financial advice. This tour is for educational purposes only.",
    "That concludes the brief tour! Feel free to ask me specific questions about any of these topics or anything else related to Bitcoin. Just type your question below.",
]

# --- Curated Resource Links (Keep it) ---
RESOURCE_LINKS = {
    r'\b(what is bitcoin|intro to bitcoin|bitcoin explained)\b': [
        {"text": "Bitcoin.org - Getting Started", "url": "https://bitcoin.org/en/getting-started"},
        {"text": "CoinDesk - What is Bitcoin?", "url": "https://www.coindesk.com/learn/what-is-bitcoin"},
    ],
    r'\b(blockchain)\b': [
        {"text": "Investopedia - Blockchain Explained", "url": "https://www.investopedia.com/terms/b/blockchain.asp"},
        {"text": "Bitcoin.org - Blockchain (Developer Guide)", "url": "https://bitcoin.org/en/developer-guide#block-chain"},
    ],
    r'\b(mining|bitcoin mining)\b': [
         {"text": "Coinbase - What is Bitcoin Mining?", "url": "https://www.coinbase.com/learn/crypto-basics/what-is-mining"},
    ],
    r'\b(satoshi nakamoto|whitepaper)\b': [
         {"text": "Bitcoin Whitepaper (Original PDF)", "url": "https://bitcoin.org/bitcoin.pdf"},
    ],
    r'\b(halving|bitcoin halving)\b': [
         {"text": "CoinGecko - Bitcoin Halving Guide", "url": "https://www.coingecko.com/learn/bitcoin-halving-guide"},
    ],
    r'\b(wallet|wallets|bitcoin wallet)\b': [
        {"text": "Bitcoin.org - Choose your Wallet", "url": "https://bitcoin.org/en/choose-your-wallet"},
    ],
    r'\b(lightning network|layer 2)\b': [
        {"text": "Lightning Network Whitepaper (Original PDF)", "url": "https://lightning.network/lightning-network-paper.pdf"}, # More technical
        {"text": "Coinbase - What is the Lightning Network?", "url": "https://www.coinbase.com/learn/crypto-basics/what-is-the-lightning-network"},
    ],
    r'\b(price|market cap|volume|coin gecko|coingecko)\b': [
         {"text": "CoinGecko - Bitcoin Price Chart", "url": "https://www.coingecko.com/en/coins/bitcoin"},
    ],
    r'\b(transaction|transactions|fees)\b': [
         {"text": "Mempool.space (Bitcoin Explorer)", "url": "https://mempool.space/"},
    ],
    # Add more keywords and relevant links here!
}

# --- Helper to find relevant links (Keep it) ---
def find_relevant_links(text_to_analyze, links_dict):
    """
    Analyzes text for keywords and returns a set of unique relevant links.
    Analyzes the provided text (usually bot response).
    """
    found_links = set()
    lower_text = text_to_analyze.lower()

    for pattern, links in links_dict.items():
        if re.search(pattern, lower_text):
            logger.debug(f"Matched pattern '{pattern}' in text.")
            for link_info in links:
                found_links.add((link_info['text'], link_info['url']))

    return list(found_links)

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Limiter Initialization (if you added it, keep it) ---
# from flask_limiter import Limiter
# from flask_limiter.util import get_remote_address
# limiter = Limiter(
#     get_remote_address,
#     app=app,
#     default_limits=["200 per day", "50 per hour"],
#     storage_uri="memory://"
# )


# --- API Endpoints ---

# Endpoint for handling user chat questions
@app.route('/ask', methods=['POST'])
# Apply rate limit if you added the feature
# @limiter.limit("10/minute")
def ask_bitcoin_api():
    """Handles POST requests with JSON data for standard chat questions."""
    if model is None:
         logger.error("Gemini model not initialized.")
         return jsonify({"answer": "The chatbot backend is not fully initialized. Please try again later or contact the administrator."}), 503

    logger.info("Received /ask request.")
    try:
        data = request.get_json()

        if not data or 'question' not in data or 'history' not in data:
            logger.warning("Invalid request: Missing 'question' or 'history' in JSON body.")
            return jsonify({"error": "Invalid request. Please provide 'question' and 'history' in the JSON body."}), 400

        user_question = data['question']
        chat_history_frontend = data['history']

        if not user_question:
             logger.info("Received empty user question.")
             return jsonify({"answer": "Please enter a question."})

        logger.info(f"User Question: {user_question}")
        logger.debug(f"Chat History received ({len(chat_history_frontend)} turns): {chat_history_frontend}")

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

        except Exception as e:
            logger.exception("An error occurred during Gemini send_message:")
            bot_response = f"An internal error occurred while contacting the AI: {e}"

        # --- Find and Append Relevant Links ---
        # Analyze the *final* bot response text for keywords
        relevant_links = find_relevant_links(bot_response, RESOURCE_LINKS)
        # Optional: Also analyze the user question
        # relevant_links.extend(find_relevant_links(user_question, RESOURCE_LINKS))
        # relevant_links = list(set(relevant_links)) # Deduplicate if combining

        if relevant_links:
             logger.info(f"Found {len(relevant_links)} relevant links.")
             # Format links in Markdown list
             links_section = "\n\n**Related Resources:**\n"
             # Sort links alphabetically by text for consistency (optional)
             relevant_links.sort(key=lambda x: x[0])
             for text, url in relevant_links:
                 links_section += f"- [{text}]({url})\n"
             bot_response += links_section # Append to the bot's response

        # Return the AI's chat response with optional links as JSON
        return jsonify({"answer": bot_response})

    except Exception as e:
        logger.exception("An unexpected error occurred in the /ask route handler:")
        return jsonify({"error": "An unexpected internal server error occurred."}), 500


# Endpoint for getting the introductory tour messages (Keep it separate)
@app.route('/tour', methods=['GET'])
# Apply a rate limit if you added the feature
# @limiter.limit("5 per hour")
def get_intro_tour():
    """Returns the list of introductory tour messages."""
    logger.info("Received /tour request.")
    try:
        return jsonify({"tour": INTRO_TOUR_MESSAGES}) # Return list under 'tour' key
    except Exception as e:
        logger.exception("An unexpected error occurred in the /tour route handler:")
        return jsonify({"error": "An unexpected internal server error occurred while fetching tour data."}), 500


# --- Handle Rate Limit Exceeded (if you added it) ---
# @app.errorhandler(429)
# def ratelimit_handler(e):
#     logger.warning(f"Rate limit exceeded for client {get_remote_address()}: {e.description}")
#     return jsonify({"error": f"Too many requests. Please try again later."}), 429

# --- Handle Generic Internal Server Error ---
@app.errorhandler(500)
def internal_error(error):
    logger.exception("Unhandled internal server error caught by error handler:")
    return jsonify({"error": "An unexpected internal server error occurred."}), 500


# --- Run the Flask App ---
if __name__ == '__main__':
    # In production (like Render), a WSGI server like Gunicorn runs the app,
    # which typically handles logging differently or relies on stderr/stdout.
    # The logging configuration above will work with Gunicorn as it writes to stderr.
    app.run(debug=True, port=5000)