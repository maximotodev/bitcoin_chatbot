from google import genai
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)

# Configure Gemini API
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_bot_response(prompt):
    try:
        response = client.models.generate_content(
    model="gemini-2.0-flash", 
    contents="You are a helpful assistant that specializes in Bitcoin. "
    "Answer questions clearly and concisely for a general audience.\n\n"
)
        reply = response.text.strip()
        logging.info(f"User: {prompt}\nBot: {reply}")
        return reply
    except Exception as e:
        logging.error("Gemini API error", exc_info=True)
        return "Sorry, I couldn’t get a response right now. Please try again soon."
