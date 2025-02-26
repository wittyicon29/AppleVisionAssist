import streamlit as st
from langchain_core.messages import HumanMessage
import os
from data_loading import load_data
from processing import process_data
from model_initialization import initialize_model, initialize_llm
from langchain.docstore.document import Document

def format_source(source):
    """Format the source information for better readability."""
    metadata = source.metadata
    title = metadata.get('title', 'Unknown Title')
    source_id = metadata.get('source', 'Unknown Source')

    if source_id and not source_id.startswith("http"):
        url = f"https://www.youtube.com/watch?v={source_id}"
    else:
        url = source_id 

    return f"**Title**: {title}\n**URL**: [Watch here]({url})\n"



def display_chat_history(chat_history):
    """Display the chat history."""
    if not chat_history:
        st.warning("No chat history to display.")
        return

    st.subheader("Chat History")
    for i, msg in enumerate(chat_history):
        if isinstance(msg, HumanMessage):
            st.markdown(f"**Query {i//2 + 1}:** {msg.content}")
        else:
            st.markdown(f"**Answer {i//2 + 1}:** {msg}")

def summarize_conversation(chat_history, llm_chain):
    """Generate a summary of the conversation using the LLM."""
    if not chat_history:
        return "No conversation to summarize."

    summary_prompt = "Summarize the following conversation:\n\n"
    conversation = ""
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            conversation += f"User: {msg.content}\n"
        else:
            conversation += f"Assistant: {msg}\n"

    input_text = summary_prompt + conversation
    doc = Document(page_content=input_text)
    summary_result = llm_chain.invoke({"context": [doc]})

    # Debugging: Log the content of summary_result
    st.write("Summary Result:", summary_result)

    if isinstance(summary_result, dict) and "answer" in summary_result:
        return summary_result["answer"]
    else:
        return summary_result

st.title("AppleVisionAssist")
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter your GROQ Cloud API key", type="password")

model_name = st.sidebar.selectbox(
    "Select Model",
    options=[
        "llama3-8b-8192",
        "llama-3.1-8b-instant",
        "llama3-groq-8b-8192-tool-use-preview",
        "gemma2-9b-it"
    ]
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.sidebar.success("Chat history cleared.")

if api_key:
    os.environ["GROQ_API_KEY"] = api_key
    st.sidebar.success("API key set successfully")

    if "docs_all" not in st.session_state:
        st.session_state.docs_all = load_data()
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = process_data(st.session_state.docs_all)
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = initialize_model(st.session_state.vectorstore, api_key, model_name)
    if "llm_chain" not in st.session_state:
        st.session_state.llm_chain = initialize_llm(api_key, model_name)

    query = st.text_input("Enter your query:")

    if st.button("Ask"):
        if query:
            result = st.session_state.rag_chain.invoke({"input": query, "chat_history": st.session_state.chat_history})
            answer = result["answer"]
            context = result["context"]

            st.markdown(f"**Answer:** {answer}")

            st.markdown("**Source(s):**")
            displayed_urls = set()
            for doc in context:
                metadata = doc.metadata
                url = metadata.get('source', 'Unknown Source')
                if url not in displayed_urls:
                    st.markdown(format_source(doc))
                    displayed_urls.add(url)

            st.session_state.chat_history.extend([HumanMessage(content=query), answer])

    if st.sidebar.button("Display Chat History"):
        display_chat_history(st.session_state.chat_history)

    if st.sidebar.button("Summarize Conversation"):
        if "llm_chain" in st.session_state:
            summary = summarize_conversation(st.session_state.chat_history, st.session_state.llm_chain)
            st.markdown(f"**Conversation Summary:** {summary}")
        else:
            st.warning("LLM chain is not initialized. Please set up the model.")

else:
    st.warning("Please enter your API key in the sidebar to continue.")
