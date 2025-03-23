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
def find_places(where_to_find: str, type_of_place_to_find_in_singular_form: str) -> str:
    """use this function to find nearby places of interest."""
    # Im in dha phase 6, lahore. find good restaurants nearby
    print("\nFIND_PLACES TOOL CALLED-X-X-X-X\n")
    import googlemaps
    GMAPS_API_KEY = secrets("GMAPS_API_KEY")
    gmaps = googlemaps.Client(key=GMAPS_API_KEY)
    geocode_result = gmaps.geocode(where_to_find)
    location = geocode_result[0]['geometry']['location']
    places_result = gmaps.places_nearby(location=location, type=type_of_place_to_find_in_singular_form, radius=1000)["results"]
  
        for place in places_result:
            details = gmaps.place(place_id=place['place_id'])['result']

            # Print additional details
            print(f"Number of Reviews: {details.get('user_ratings_total', 'N/A')}")
            if 'opening_hours' in details:
                print("Business Hours:")
                for day in details['opening_hours'].get('weekday_text', []):
                    print(day)
            else:
                print("Business Hours not available.")
                
            print("\n")  # Add space between different places
            
    return ("Following is a list of relevant places I found. Prepare a new list containing only those places whose address includes "
            f"{where_to_find}: {places_result}"
            )

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