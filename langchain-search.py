from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore', message='.*shadows an attribute in parent.*')

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")

# Use create_agent without custom prompt - it handles tool usage automatically
agent = create_agent(model=llm, tools=tools)

def main():
   result = agent.invoke({"messages": [HumanMessage(content="Search for engineering manager in AI & ML 3 job posting in California using langchain")]})
   print(result)
   
if __name__ == "__main__":
    main()