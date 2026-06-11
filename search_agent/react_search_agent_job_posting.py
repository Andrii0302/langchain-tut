from dotenv import load_dotenv
import os

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

if not os.environ.get("GOOGLE_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
    raise ValueError(
        "Make sure both GOOGLE_API_KEY and TAVILY_API_KEY are set in your environment/.env file."
    )
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
tools = [TavilySearch(max_results=5)]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Started")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="Search for 3 job posting for an AI engineer using langchain in the bay area on linkedin and list their details"
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()
