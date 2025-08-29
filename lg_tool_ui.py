# %%
import os
from dotenv import load_dotenv
import bs4
import streamlit as st
from typing_extensions import List, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langchain_core.messages import SystemMessage, HumanMessage

# --- Configuration and Setup ---

def load_model_and_embeddings():
    load_dotenv()
    secret_key = os.getenv('OPENAI_API_KEY')
    model = init_chat_model("gpt-4o-mini", model_provider="openai", api_key=secret_key)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return model, embeddings

def load_and_index_documents(embeddings):
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=all_splits)
    print(f"Indexed {len(all_splits)} chunks.")
    return vector_store

# --- Tool and Graph Definitions ---

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    model_with_tools = model.bind_tools([retrieve])
    response = model_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
    )
    # Only add docs_content if it exists
    if docs_content.strip():
        system_message_content += "\n\n" + docs_content

    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = model.invoke(prompt)
    return {"messages": [response]}

def build_graph():
    graph_graph_builder = StateGraph(MessagesState)
    graph_graph_builder.add_node(query_or_respond)
    graph_graph_builder.add_node(ToolNode([retrieve]))
    graph_graph_builder.add_node(generate)
    graph_graph_builder.set_entry_point("query_or_respond")
    graph_graph_builder.add_conditional_edges(
        "query_or_respond", tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_graph_builder.add_edge("tools", "generate")
    graph_graph_builder.add_edge("generate", END)
    return graph_graph_builder.compile()


def streamlit_chat(graph, messages):
    # Display all previous messages
    for message in messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            # Only display assistant messages with non-empty content
            if message["content"] and message["content"].strip():
                st.chat_message("assistant").write(message["content"])

    # Get new user input
    prompt = st.chat_input("What's your question?")
    if prompt:
        # Show user message immediately
        st.chat_message("user").write(prompt)
        messages.append({"role": "user", "content": prompt})

        # Process and show assistant response
        response = graph.invoke({
            "messages": [HumanMessage(content=prompt)]
        })
        for message in response["messages"]:
            # Only display assistant messages with non-empty content
            if message.content and message.content.strip():
                st.chat_message("assistant").write(message.content)
            messages.append({"role": "assistant", "content": message.content})

def display_chatbot(graph):
    st.title("Agentic AI Chatbot")
    st.write("Ask questions about the blog post on agentic AI.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    streamlit_chat(graph, st.session_state.messages)
# --- Main Execution ---

if __name__ == "__main__":
    st.set_page_config(page_title="Agentic AI Chatbot", page_icon=":robot:")
    model, embeddings = load_model_and_embeddings()
    global vector_store
    vector_store = load_and_index_documents(embeddings)
    graph = build_graph()
    display_chatbot(graph)



