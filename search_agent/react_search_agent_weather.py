from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """Tool that searches over Internet
    Args:
        query: The query to search for
    Returns:
        The search result"""
    print(f"Searching for {query}")
    return tavily.search(query=query)


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Started")
    result = agent.invoke(
        {"messages": HumanMessage(content="What is the weather in Tokyo")}
    )
    print(result)


if __name__ == "__main__":
    main()
