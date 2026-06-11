import os
from typing import List

from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

if not os.environ.get("GOOGLE_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
    raise ValueError(
        "Make sure both GOOGLE_API_KEY and TAVILY_API_KEY are set in your environment/.env file."
    )


class Source(BaseModel):
    """Schema for a source used by the agent."""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources."""

    answer: str = Field(description="The Agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list,
        description="The list of sources used to generate the answer.",
    )


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
tools = [TavilySearch(max_results=5)]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Started")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="Search for 3 job postings for an AI engineer using langchain in the bay area on linkedin and list their details"
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()
