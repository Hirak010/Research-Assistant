from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import uvicorn
from langchain_core.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

MODEL_NAMES = [
    "llama3-70b-8192",
    "mixtral-8x7b-32768"
]

tool_tavily = TavilySearchResults(max_results=1)
tools = [tool_tavily]

app = FastAPI(title="Langgraph AI Agent")

class RequestState(BaseModel):
    model_name: str
    system_prompt: str
    messages: str

@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API endpoint to interact with the chatbot using LangGraph and tools.
    Dynamically selects the model specified in the request.
    """
    if request.model_name not in MODEL_NAMES:
        return {"error": "Model not found."}
    
    llm = ChatGroq(
        model=request.model_name,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    
    # # Append custom instructions for tool usage to the system prompt.
    # custom_instructions = (
    #     "When you need to perform a web search using the tool, output only a valid JSON object in the following format: "
    #     "{\"query\": \"<your search query>\"}. Do not include any additional text or formatting."
    # )
    # final_system_prompt = request.system_prompt + "\n" + custom_instructions
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", request.system_prompt),
        ("placeholder", "{request.messages}"),
    ]
)
    # Create the agent without the prompt_template argument.
    agent = create_react_agent(llm, tools=tools, prompt=prompt)

    #state = {"messages": request.messages}
    result = agent.invoke({"messages": [("human", request.messages)]})    

    return result

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
