from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import AgentExecutor, create_react_agent
from langsmith import Client
client = Client()

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
react_prompt = client.pull_prompt("hwchase17/react", include_model=True)

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor
def main():
   result = chain.invoke({"input": "Search for engineering manager in AI & ML 3 job posting in Californa using langchain"})
   print(result)
   
if __name__ == "__main__":
    main()