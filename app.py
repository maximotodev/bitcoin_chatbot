import os
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests # Import the requests library
import time # Import time for last updated timestamp formatting

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
    # model.generate_content("test", stream=True) # Removed test for speed during dev
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error creating or testing Gemini model '{GEMINI_MODEL_NAME}': {e}")
    print("Please check your GEMINI_MODEL_NAME in the .env file and ensure it's valid for generateContent.")
    exit()

# --- System Prompt: Define the Chatbot's Persona and Scope ---
# IMPORTANT: Update the prompt to inform the model about the real-time data it receives
SYSTEM_PROMPT = """
You are a specialized Bitcoin chatbot. Your knowledge base is focused ONLY on Bitcoin.
Answer the user's questions accurately and concisely about Bitcoin concepts, history,
technology (blockchain, mining, cryptography), economic aspects (market trends, adoption),
notable figures, significant events, and related technical details (like forks, wallets, Layer 2 solutions like Lightning Network).

You are also provided with RECENT, REAL-TIME Bitcoin data (price, market cap, volume, etc.) at the beginning of the user's input.
**Use this provided real-time data if the user asks questions about the current price, market cap, or trading volume.**
If the user asks about historical prices or trends, you can use your general knowledge but make it clear you are not using the *provided* real-time figure.
If the user asks for data *not* included in the provided real-time information, use your general knowledge if possible or state you only have access to the provided data points.

- Be factual and objective.
- Do NOT give financial advice, investment recommendations, or price predictions beyond interpreting the provided real-time data when asked specifically about *current* values.
- If a question is clearly unrelated to Bitcoin (e.g., asking about Ethereum details, other cryptocurrencies unless in direct comparison to Bitcoin, cooking recipes, or general knowledge), politely state that you specialize in Bitcoin and cannot answer the unrelated query.
- Keep answers focused and avoid unnecessary jargon, but explain technical terms if needed.
"""

# --- Helper to fetch real-time data from CoinGecko ---
def get_bitcoin_data():
    """Fetches current Bitcoin price, market cap, volume from CoinGecko."""
    # CoinGecko API endpoint for basic data
    api_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_market_cap=true&include_24hr_vol=true&include_last_updated_at=true"

    try:
        response = requests.get(api_url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if 'bitcoin' in data and 'usd' in data['bitcoin']:
            btc_data = data['bitcoin']
            price = btc_data.get('usd')
            market_cap = btc_data.get('usd_market_cap')
            volume_24h = btc_data.get('usd_24h_vol')
            last_updated_unix = btc_data.get('last_updated_at') # Unix timestamp

            last_updated_readable = "N/A"
            if last_updated_unix:
                 try:
                    # Convert unix timestamp to readable format
                    last_updated_readable = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(last_updated_unix))
                 except Exception:
                     pass # Ignore formatting errors

            # Format the data clearly for the prompt
            formatted_data = "--- REAL-TIME BITCOIN DATA ---\n"
            formatted_data += f"Current Price (USD): ${price:,.2f}\n" if price is not None else "Current Price (USD): N/A\n"
            formatted_data += f"Market Cap (USD): ${market_cap:,.2f}\n" if market_cap is not None else "Market Cap (USD): N/A\n"
            formatted_data += f"24h Volume (USD): ${volume_24h:,.2f}\n" if volume_24h is not None else "24h Volume (USD): N/A\n"
            formatted_data += f"Last Updated (UTC): {last_updated_readable}\n"
            formatted_data += "--------------------------\n\n"

            return formatted_data, None # Return data string and no error

        else:
            return "Could not parse Bitcoin data from API.", "API response format unexpected."

    except requests.exceptions.RequestException as e:
        # Handle network errors, timeouts, HTTP errors
        print(f"Error fetching Bitcoin data from CoinGecko: {e}")
        return "Could not fetch real-time data due to an API error.", f"Request Error: {e}"
    except Exception as e:
        # Handle other potential errors
        print(f"An unexpected error occurred fetching Bitcoin data: {e}")
        return "Could not fetch real-time data due to an unexpected error.", f"Unexpected Error: {e}"


# --- Helper to format history for Gemini API ---
def format_history_for_gemini(history):
    """Converts history from frontend format to Gemini API format."""
    gemini_history = []
    # Prepend the system prompt as the initial 'user' turn
    gemini_history.append({'role': 'user', 'parts': [{'text': SYSTEM_PROMPT}]})
    # The very first expected response from the model after the system prompt
    # is typically empty or a greeting, which we represent here.
    gemini_history.append({'role': 'model', 'parts': [{'text': 'Okay, I am ready to answer questions about Bitcoin.'}]})

    # Add previous chat turns
    for message in history:
        role = 'user' if message['type'] == 'user' else 'model'
        # Important: When sending history, do NOT include the real-time data block again.
        # Only the *current* user turn includes it.
        # So, we just add the message text.
        # A more robust approach might involve stripping the data block if it was part of a previous turn,
        # but since we add it only to the *current* turn's input to send_message, this is simpler.
        gemini_history.append({'role': role, 'parts': [{'text': message['text']}]})

    return gemini_history

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- API Endpoint ---

@app.route('/ask', methods=['POST'])
def ask_bitcoin_api():
    """Handles POST requests with JSON data and returns a JSON response."""
    data = request.get_json()

    if not data or 'question' not in data or 'history' not in data:
        return jsonify({"error": "Invalid request. Please provide 'question' and 'history' in the JSON body."}), 400

    user_question = data['question']
    chat_history_frontend = data['history']

    if not user_question:
         return jsonify({"answer": "Please enter a question."})

    # --- Fetch Real-time Data ---
    realtime_data_string, fetch_error = get_bitcoin_data()

    # --- Prepare Prompt for Gemini ---
    # Combine real-time data with the user's question for the *current* turn
    # The AI is instructed in the SYSTEM_PROMPT to expect this format.
    # This data is *not* added to the chat history that's passed to start_chat.
    prompt_with_data = f"{realtime_data_string}\nUser Question: {user_question}\n\nAnswer:"

    # If fetching data failed, inform the user within the response
    if fetch_error:
        # Prepend the error message to the AI's potential answer or handle separately
        # For simplicity, we'll let the AI respond, but inform it about the data issue
        # A better approach might be to send a simpler prompt if data fetch fails badly
        print(f"Warning: Data fetch error: {fetch_error}") # Log the error
        # We could modify prompt_with_data or add an instruction for this turn
        # Let's add a note within the data block indicating the failure
        prompt_with_data = f"--- REAL-TIME BITCOIN DATA ---\nData Fetch Failed: {fetch_error}\n--------------------------\n\nUser Question: {user_question}\n\nAnswer:"


    # --- Generate Content ---
    bot_response = "Sorry, I couldn't process your request."

    try:
        # Use start_chat with the history (excluding the current turn's real-time data block)
        gemini_history = format_history_for_gemini(chat_history_frontend)
        chat = model.start_chat(history=gemini_history)

        # Send the *current* user question along with the real-time data string
        response = chat.send_message(prompt_with_data)

        # Process the response
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

# --- Run the Flask App ---
if __name__ == '__main__':
    # Ensure the venv is activated before running
    app.run(debug=True, port=5000)