from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

class Source(BaseModel):
    """Schema for a source used by the agent."""
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the agent's response."""
    answer: str = Field(description="The answer to the user's question")
    sources: List[Source] = Field(description="The sources used to answer the question")

load_dotenv()

llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    result = agent.invoke({"messages": HumanMessage(content="Search for engineering manager in AI & ML 3 job posting in Californa using langchain")})
    print(result)
if __name__ == "__main__":
    main()