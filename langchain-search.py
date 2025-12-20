from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain import hub
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor

load_dotenv()

llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    result = agent.invoke({"messages": HumanMessage(content="Search for engineering manager in AI & ML 3 job posting in Californa using langchain")})
    print(result)