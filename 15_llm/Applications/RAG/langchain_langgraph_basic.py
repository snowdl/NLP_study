from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage

# Define the state type using TypedDict and Annotated for LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the graph builder with the defined state
graph_builder = StateGraph(State)

# Sample messages to test the behavior of add_messages
msg1 = [HumanMessage(content="Welcome to my NLP studyGithub!", id="1")]
msg2 = [AIMessage(content="Nice to see you~", id="2")]
msg3 = [HumanMessage(content="Hello", id="1")]
msg4 = [HumanMessage(content="Nice to see you", id="1")]

# Merge messages
result1 = add_messages(msg1, msg2)
result2 = add_messages(msg3, msg4)

# Print results
print('==== result1 ====')
print(result1)
print("\n")
print('==== result2 ====')
print(result2)

'''
Expected output:

==== result1 ====
[HumanMessage(content='Welcome to my NLP studyGithub!', additional_kwargs={}, response_metadata={}, id='1'),
 AIMessage(content='Nice to see you~', additional_kwargs={}, response_metadata={}, id='2')]

==== result2 ====
[HumanMessage(content='Nice to see you', additional_kwargs={}, response_metadata={}, id='1')]
'''
