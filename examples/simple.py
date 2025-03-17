import asyncio

from dotenv import load_dotenv
from custom_openai_client import CustomAzureOpenAI

from browser_use import Agent

load_dotenv()

# Initialize the model
llm = CustomAzureOpenAI(
	deployment_name="gpt-4o",
	model="gpt-4o",
	api_version="2024-10-21"
)
task = 'Find the founders of browser-use and draft them a short personalized message'

agent = Agent(task=task, llm=llm)


async def main():
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())
