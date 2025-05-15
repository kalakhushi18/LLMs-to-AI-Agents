from dotenv import load_dotenv  #loadin env file
from pydantic import BaseModel  #loading structured output 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser  #this will tell llm to generate propmt response in this format
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import save_tool, search_tool, wiki_tool


load_dotenv()

#creating parser class
class ResearchResponse(BaseModel):  #all the fileds in output
    topic : str
    summary : str
    sources : list[str]
    tools_used : list[str]

#Setting up an LLM

llm = ChatOpenAI(model = "gpt-3.5-turbo")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

#prompt template 

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
#Agent 
agent = create_tool_calling_agent(
    llm= llm, 
    tools= tools,
    prompt= prompt
)

agent_executor = AgentExecutor(agent=agent, tools= tools, verbose=True)
query = input("What can I help you research? ")
raw_response = agent_executor.invoke({"query": query})

#creating structure of output
try:
    structured_response = parser.parse(raw_response.get("output"))
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)



#first check of llm
# response = llm.invoke("What is meaning of Daylight saving?")
# print(response)

#creating prompt template (structured output)






