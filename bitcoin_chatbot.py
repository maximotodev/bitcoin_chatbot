import os
import google.generativeai as genai
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv() # Load environment variables from .env file
API_KEY = os.getenv("GOOGLE_API_KEY")

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
# Choose a model (e.g., 'gemini-2.0-flash-001')
# See available models: https://ai.google.dev/models/gemini
try:
    model = genai.GenerativeModel('gemini-2.0-flash-001')
except Exception as e:
    print(f"Error creating Gemini model: {e}")
    exit()
# --- Add this code temporarily ---
print("Listing available models supporting 'generateContent'...")
supported_models = [
    m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods
]
if not supported_models:
    print("No models found that support 'generateContent'. Please check your API key and region.")
else:
    print("Available models supporting 'generateContent':")
    for m in supported_models:
        print(f"- {m.name}")
print("-" * 20)
# --- End temporary code ---
# --- System Prompt: Define the Chatbot's Persona and Scope ---
# This guides the AI to focus on Bitcoin and act knowledgeable.
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

# --- Chat History (Optional but Recommended for Context) ---
# For a simple single-turn Q&A, we don't strictly need this,
# but it's good practice for more conversational bots.
# We will prepend the system prompt to each user query for simplicity here.
# For multi-turn conversations, use model.start_chat(history=[...])

print("--- Bitcoin Chatbot Initialized ---")
print("Ask me anything about Bitcoin! Type 'quit', 'exit', or 'bye' to end.")

# --- Main Chat Loop ---
while True:
    user_input = input("\nYou: ").strip()

    if not user_input:
        continue # Ask again if input is empty

    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("Chatbot: Goodbye!")
        break

    # Combine the system prompt with the user's current question
    # This ensures the model stays focused on its Bitcoin persona for each query.
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser Question: {user_input}\n\nAnswer:"

    print("Chatbot: Thinking...")

    try:
        # --- Generate Content ---
        response = model.generate_content(
            full_prompt,
            # Optional: Configure safety settings, temperature, etc.
            # generation_config=genai.types.GenerationConfig(...)
            # safety_settings=[...]
            )

        # --- Display Response ---
        # Access the text part of the response
        # Handle cases where the response might be blocked or empty
        if response.parts:
             # Clean up potential markdown formatting if not desired
            bot_response = response.text.strip()
            print(f"\nChatbot: {bot_response}")
        else:
             # Check prompt_feedback for reasons like safety blocks
             block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
             print(f"\nChatbot: I couldn't generate a response for that. Reason: {block_reason}")
             if response.candidates and response.candidates[0].finish_reason != 'STOP':
                print(f"       Finish Reason: {response.candidates[0].finish_reason}")


    except Exception as e:
        print(f"\nChatbot: An error occurred while contacting the Gemini API: {e}")
        # Consider adding more specific error handling based on Gemini API exceptions