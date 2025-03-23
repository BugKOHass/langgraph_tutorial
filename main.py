from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from langgraph.types import Command, interrupt

import os
import json
import requests

from decouple import Config, RepositoryEnv


secrets = Config(RepositoryEnv(".env"))

TAVILY_API_KEY = secrets("TAVILY_API_KEY")
OPENAI_API_KEY = secrets("OPENAI_API_KEY")

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
# prepare profile of journalist named nfp
@tool
def journalist_profile(name_of_journalist: str) -> str:
    """Prepare the profile of a given journalist."""
    print("\nJOURNALIST_PROFILE TOOL CALLED-X-X-X-X\n")
    from fetch import fetch_articles
    from process import process_article
    if name_of_journalist != "nfp": return "Sorry, requested profile is not available. Here's a lollypop for you!"
    articles_urls = fetch_articles()
    analyses = "".join([process_article(url["url"]) for url in articles_urls])
    return ("Prepare the profile of the journalist by analyzing these articles; your evaluation should be only one sentence long;"
            "it should not be a summary of these articles but you have to deduce the hidden beliefs/agendas of the journalist from "
            f"these articles: {analyses}"
            )

# find nearby places of interest
@tool
def find_places(where_to_find: str, type_of_place_to_find: str) -> str:
    """use this function to find nearby places of interest."""
    # Im in dha phase 6, lahore. find good restaurants nearby
    print("\nFIND_PLACES TOOL CALLED-X-X-X-X\n")
    GOOGLE_API_KEY = secrets("GOOGLE_API_KEY")
    GOOGLE_SEARCH_ENGINE_ID = secrets("GOOGLE_SEARCH_ENGINE_ID")

    url = 'https://www.googleapis.com/customsearch/v1'
    query = f"{type_of_place_to_find} near {where_to_find}. Mention their reviews, ratings, business hours and cuisine."
    # Set parameters for the request
    params = {
        'key': GOOGLE_API_KEY,
        'cx': GOOGLE_SEARCH_ENGINE_ID,
        'q': query,
    }
    
    # Make a GET request to fetch search results
    response = requests.get(url, params=params)
    
    # Check if request was successful
    if response.status_code == 200:
        return ("Following is a list of places I found. For each place, extract the following details: name, address, number of reviews, rating, business hours."
                f"{response.json()}"
        )
    else:
        return "server error"


os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
tool = TavilySearchResults(max_results=2)
tools = [tool, journalist_profile, find_places]

openai_api_key = OPENAI_API_KEY
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    api_key=openai_api_key
)

llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    return {"messages": [message]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception as e:
#     print(f"error: {str(e)}")

config = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    system_msg = {"role": "system", "content": "You respond in one sentence only."}
    user_msg = {"role": "user", "content": user_input}
    events = graph.stream(
        {"messages": [user_msg]},
        config,
        stream_mode="values"
        )
    for event in events:
        event["messages"][-1].pretty_print()

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    stream_graph_updates(user_input)