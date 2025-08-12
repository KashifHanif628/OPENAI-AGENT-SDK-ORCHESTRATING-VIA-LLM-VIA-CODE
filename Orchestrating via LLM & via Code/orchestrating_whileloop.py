"""Orchestrating via Code with while loop"""

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
    print("=== AI Orchestrator ===")
    print("Type 'exit' anytime to quit.\n")

    while True:
        # Step 1: Ask for text first
        user_text = input("Apna text dijiye: ").strip()

        # Exit option
        if user_text.lower() == "exit":
            print("Program band ho gaya. Shukriya! 👋")
            break

        # Step 2: Ask for action
        action = input("Aap kya karna chahte hain? (translate / summarize): ").strip().lower()

        # Exit option
        if action == "exit":
            print("Program band ho gaya. Shukriya! 👋")
            break

        # Step 3: Decide which agent to run
        if action == "translate":
            selected_agent = translator_agent
        elif action == "summarize":
            selected_agent = summarizer_agent
        else:
            print("❌ Galat option. Sirf 'translate' ya 'summarize' likhein.\n")
            continue

        # Step 4: Run selected agent
        result = await Runner.run(
            selected_agent,
            input=user_text,
            run_config=config
        )

        # Step 5: Show output
        print("\n=== Final Output ===")
        print(result.final_output)
        print("\n--------------------------------\n")

if __name__ == "__main__":
    asyncio.run(main())


"""
Orchestrating via Code (with While Loop)

Decision Control in Code – Kaunsa agent run hoga (Translator ya Summarizer) yeh decision puri tarah Python code ke 
if-else logic me hota hai, na ke LLM ke andar.

LLM as Task Performer – LLM sirf apna specialized kaam karta hai, jaise translation ya summarization, jo code se 
select kiya gaya ho.

Loop for Continuous Interaction – while loop user ko ek hi session me multiple requests run karne ka option deta hai, 
bina program dobara start kiye.

Clear Separation of Roles – Code decision-making handle karta hai, aur LLM execution handle karta hai, jis se flow 
predictable aur maintainable rehta hai.

"""