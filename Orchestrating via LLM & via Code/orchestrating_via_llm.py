"""Orchestrating via LLM & For loop"""
"""AI Q&A with Answer Improvement in For Loop"""

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

# Q&A Agent
qa_agent = Agent(
    name="qa_agent",
    instructions="Answer the user's question as accurately and clearly as possible.",
    model=external_model
)

# Run configuration
config = RunConfig(
    model=external_model,
    model_provider=external_client,
    tracing_disabled=True
)

async def main():
    print("=== AI Q&A Session ===")
    try:
        total_runs = int(input("how many times would you like to run ?: ").strip())
    except ValueError:
        print("❌ please write in numbers.")
        return

    for i in range(total_runs):
        print(f"\n--- Question {i+1} ---")
        
        # Step 1: Ask question
        user_question = input("Please ask your question: ").strip()

        # Step 2: Get initial answer from agent
        result = await Runner.run(
            qa_agent,
            input=user_question,
            run_config=config
        )

        answer = result.final_output
        print("\nAgent respond:")
        print(answer)

        # Step 3: Give user a chance to ask for improvement
        while True:
            feedback = input("\n Is answer according to you? (yes / no): ").strip().lower()
            if feedback == "yes":
                break
            elif feedback == "no":
                # Ask agent to improve the answer
                improvement_prompt = f"Improve the following answer to be clearer and more accurate:\n{answer}"
                result = await Runner.run(
                    qa_agent,
                    input=improvement_prompt,
                    run_config=config
                )
                answer = result.final_output
                print("\nImproved Answer:")
                print(answer)
            else:
                print("❌ Please reply in 'yes' or 'no'.")

    print("\n=== Session Finished ===")

if __name__ == "__main__":
    asyncio.run(main())



"""
1- Orchestrating via LLM in simple-way.

Kisi bhi application ka execution flow ek LLM control karta hai.
LLM decide karta hai ke kaunsa tool, kaunsa sub-agent ya handoff pehle chalana hai, aur baad me kaunsa.
Achha orchestration ke liye prompt design aur tools ka clear documentation (docstrings) bohot important hai.


Flow Structure:
Main Agent → user query receive karta hai.
Sub-Agents → specialized tasks ke liye kaam karte hain.
Function Tools → specific operations perform karte hain.
Handoff → ek agent se dusre agent ko query transfer hoti hai.
LLM → poore process ko step-by-step control karta hai.


Monitoring the Flow
Tracing: Production me use hota hai taake backend me kaunsa step kaise execute hua, track ho sake.
Verbose Mode: Development me debugging aur detailed logs ke liye use hota hai.


Example: Multiple LLM Usage
User ek sawal puchta hai → Agent answer deta hai.
Agar answer user ko pasand nahi aata → User re-generation ka request deta hai.


Improvement Loop:
Improvement Agent answer check karta hai.
Agar answer theek hai → user ko forward kar deta hai.
Agar answer galat hai → main agent ko better answer generate karne ka kehta hai.
Yeh loop 1 se 5 ya 10 iterations tak chal sakta hai taake har answer progressively better ho.

"""