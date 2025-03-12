from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Import the necessary components from your existing code
from langgraph.graph import StateGraph
from typing import TypedDict, List
from langchain_community.tools import TavilySearchResults
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")



llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

# Define structured state
class ArticleState(TypedDict):
    subject: str
    content_details: str
    revised: str  # Holds the latest version of the article
    critique: str
    references: List[str]
    search_queries: List[str]
    external_information: List[str]
    iteration_count: int  # Tracks the number of iterations


# Initialize Tavily tool
tavily_tool = TavilySearchResults(max_results=3)

# Initialize GPT-4o as the LLM
#llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Add a rate limiter decorator
def rate_limited(max_tokens_per_minute=6000, estimated_tokens_per_call=1500):
    interval = 60 / (max_tokens_per_minute / estimated_tokens_per_call)
    def decorator(func):
        def wrapper(*args, **kwargs):
            time.sleep(interval)
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Add truncation function for long content
def truncate_content(text, max_tokens=2000):
    return ' '.join(text.split()[:max_tokens])
# 1. Generate an Initial Draft
def generate_draft(state: ArticleState) -> ArticleState:
    """Generates an initial draft of the article based on the subject and content details."""
    prompt = f"""
    Write an article on the subject: "{state['subject']}". 
    Details to cover: {state['content_details']}.

    The article should include an introduction, a well-structured body, and a conclusion.
    """
    response = llm.invoke(prompt)

    return {
        **state,
        "revised": response.content or "Initial draft content placeholder.",
        "iteration_count": 0,  # Initialize iteration counter
    }


# 2. Revise the Draft with Citations
def revise_draft(state: ArticleState) -> ArticleState:
    """Refines the article, integrating references and improving clarity."""
    prompt = f"""
    Improve the following article by making it clearer, more concise, and more accurate.
    Ensure numerical citations in [#] format and add a references section.

    Subject: {state['subject']}
    Content Details: {state['content_details']}

    Current Draft:
    {state['revised']}
    """
    response = llm.invoke(prompt)

    return {**state, "revised": response.content or state["revised"]}


# 3. Critique and Reflection
def critique_article(state: ArticleState) -> ArticleState:
    """Provides a critique of the current version of the article and suggests improvements."""
    prompt = f"""
    Critique the following article. Identify weaknesses, missing information, and unnecessary content.
    Provide three key areas for improvement.

    Subject: {state['subject']}
    Content Details: {state['content_details']}

    Article:
    {state['revised']}
    """
    response = llm.invoke(prompt)

    return {**state, "critique": response.content or "No critique available."}


# 4. Generate Research Queries
def generate_search_queries(state: ArticleState) -> ArticleState:
    """Generate 3 focused queries instead of 5"""
    prompt = f"""
    Generate 3 focused research queries to improve: "{state['subject']}".
    Focus on finding technical documentation and recent trends.
    """
    response = llm.invoke(prompt)
    queries = response.content.split("\n")[:3]  # Take only first 3
    return {**state, "search_queries": queries}


# 5. Fetch External Information
def fetch_external_information(state: ArticleState) -> ArticleState:
    """Fetch and filter external info"""
    external_info = []
    for query in state["search_queries"][:3]:  # Process only first 3 queries
        if query.strip():
            print(f"🔍 Searching for: {query}")
            try:
                results = tavily_tool.invoke(query)
                # Extract just the relevant snippets
                if results and 'results' in results:
                    external_info.extend([
                        f"Source: {res['title']}\nSnippet: {res['content'][:200]}..."
                        for res in results['results'][:2]  # Take top 2 results
                    ])
            except Exception as e:
                print(f"⚠️ Error searching {query}: {e}")
    return {**state, "external_information": external_info}

# 6. Iterative Refinement
@rate_limited()
def iterative_refinement(state: ArticleState) -> ArticleState:
    """Refines the article based on critique and external information."""
    # Truncate inputs to stay within token limits
    truncated_info = truncate_content(str(state["external_information"]))
    truncated_critique = truncate_content(state["critique"])
    truncated_draft = truncate_content(state["revised"])

    # Create the prompt
    prompt = f"""
    Update and improve the article on "{state['subject']}" using the feedback and research below.
    Ensure all claims are well-supported and properly cited.

    External Information:
    {truncated_info}

    Critique:
    {truncated_critique}

    Current Draft:
    {truncated_draft}
    """
    
    # Invoke the LLM
    response = llm.invoke(prompt)

    # Return the updated state
    return {
        **state,  # Preserve all existing state fields
        "revised": response.content,  # Update the revised content
        "iteration_count": state["iteration_count"] + 1,  # Increment counter
    }


# 7. Final Step
def final_step(state: ArticleState) -> ArticleState:
    """Finalizes the article after three iterations."""
    return {
        **state,
        "revised": state["revised"]
        + f"\n\nFinalized after 3 iterations.\n\nArticle on: {state['subject']}.",
    }


# 8. Conditional Check for Looping
def should_continue(state: ArticleState) -> str:
    """Determines whether to refine again or finish."""
    return "critique_article" if state["iteration_count"] < 1 else "final_step"







# Define the request model
class ChatRequest(BaseModel):
    subject: str
    content_details: str

# Define the response model
class ChatResponse(BaseModel):
    article: str
    iteration_count: int
    references: Optional[List[str]] = None

# Initialize FastAPI app
app = FastAPI(title="Article Generation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # Reuse your existing ArticleState and workflow setup
# class ArticleState(TypedDict):
#     subject: str
#     content_details: str
#     revised: str
#     critique: str
#     references: List[str]
#     search_queries: List[str]
#     external_information: List[str]
#     iteration_count: int

# Initialize your existing workflow
workflow = StateGraph(ArticleState)
# Add Nodes
workflow.add_node("generate_draft", generate_draft)
workflow.add_node("revise_draft", revise_draft)
workflow.add_node("critique_article", critique_article)
workflow.add_node("generate_search_queries", generate_search_queries)
workflow.add_node("fetch_external_information", fetch_external_information)
workflow.add_node("iterative_refinement", iterative_refinement)
workflow.add_node("final_step", final_step)

# Define Execution Flow
workflow.set_entry_point("generate_draft")
workflow.add_edge("generate_draft", "revise_draft")
workflow.add_edge("revise_draft", "critique_article")
workflow.add_edge("critique_article", "generate_search_queries")
workflow.add_edge("generate_search_queries", "fetch_external_information")
workflow.add_edge("fetch_external_information", "iterative_refinement")

# Add Conditional Edge for Looping
workflow.add_conditional_edges(
    "iterative_refinement",
    should_continue,
    {"critique_article": "critique_article", "final_step": "final_step"},
)

# Set Final Step
workflow.set_finish_point("final_step")
article_workflow = workflow.compile()

@app.post("/chat", response_model=ChatResponse)
async def generate_article(request: ChatRequest):
    try:
        # Prepare input for the workflow
        input_message = {
            "subject": request.subject,
            "content_details": request.content_details
        }

        # Run the workflow
        result = article_workflow.invoke(input_message)

        # Prepare the response
        response = ChatResponse(
            article=result["revised"],
            iteration_count=result["iteration_count"],
            references=result.get("references", [])
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating article: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)