from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langsmith import Client
client = Client()

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
react_prompt_template = client.pull_prompt("hwchase17/react", include_model=True)
react_prompt = react_prompt_template.template

agent = create_agent(model=llm, tools=tools, system_prompt=react_prompt)

def main():
   result = agent.invoke({"input": "Search for engineering manager in AI & ML 3 job posting in Californa using langchain"})
   print(result)
   
if __name__ == "__main__":
    main()