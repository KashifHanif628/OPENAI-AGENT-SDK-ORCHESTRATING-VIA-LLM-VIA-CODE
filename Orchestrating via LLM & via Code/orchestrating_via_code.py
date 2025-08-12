"""Orchestrating via Code"""
"""Decision making Flow via Code step by step"""

from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# External OpenAI client for Gemini API
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model setup
external_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

# Agents
translator_agent = Agent(
    name="translator_agent",
    instructions="Translate the input text into English language.",
    model=external_model
)

summarizer_agent = Agent(
    name="summarizer_agent",
    instructions="Summarize the given text in 3 short bullet points.",
    model=external_model
)

# Run configuration
config = RunConfig(
    model=external_model,
    model_provider=external_client,
    tracing_disabled=True
)

async def main():
    # Step 1: Ask user for action
    action = input("Aap kya karna chahte hain? (translate / summarize): ").strip().lower()

    # Step 2: Ask user for text
    user_text = input("Apna text dijiye: ").strip()

    # Step 3: Decide which agent to run
    if action == "translate":
        selected_agent = translator_agent
    elif action == "summarize":
        selected_agent = summarizer_agent
    else:
        print("❌ Galat option. Sirf 'translate' ya 'summarize' likhein.")
        return

    # Step 4: Run selected agent
    result = await Runner.run(
        selected_agent,
        input=user_text,
        run_config=config
    )

    # Step 5: Show output
    print("\n=== Final Output ===")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())


"""
Orchestrating via Code in Simple-Way.

Kisi bhi application ka execution flow Python code control karta hai — LLM sirf task perform karta hai, 
decision-making code ke andar hoti hai.

Code decide karta hai ke kaunsa agent pehle run hoga, aur kaunsa baad me, based on user input ya predefined logic.
Achha orchestration ke liye agents clearly defined hone chahiye, aur unka kaam code me clearly specify hona chahiye.


Flow Structure:
User Input → User se poocha jata hai ke woh translation karna chahta hai ya summarization.
Decision Logic (if-else) → Python code decide karta hai ke kaunsa agent run karna hai.
Selected Agent → Chosen agent (Translator ya Summarizer) user ka text process karta hai.
Final Output → Result user ko display hota hai.


Monitoring the Flow
Manual Control → Developer ke paas poora control hota hai ke kis situation me kaunsa agent run hoga.
Tracing / Verbose → Agar enable kiya jaye to debugging ke liye execution ka detail log rakhna possible hai.


Example: Multiple Agent Usage
User ek text deta hai → Code user se action poochta hai.
Agar user "translate" choose kare → Translator Agent run hota hai.
Agar user "summarize" choose kare → Summarizer Agent run hota hai.


Above code is perfect example in this case, because there is complete control in python hand., not in LLM contol — 
That is called "Orchestrating via Code".

"""

