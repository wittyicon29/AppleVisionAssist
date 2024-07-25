import streamlit as st
from langchain_core.messages import HumanMessage
import os
from langchain.docstore.document import Document
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, YoutubeLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq

def initialize_model(vectorstore, api_key, model_name):
    """Initialize the LLM and create the prompt template and RAG chain."""
    llm = ChatGroq(model=model_name, groq_api_key=api_key)

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, vectorstore.as_retriever(), contextualize_q_prompt
    )
    
    system_prompt = """
        You are an assistant for question-answering and sales tasks. \
        Use the following pieces of retrieved context to construct a well-structured and persuasive response. \
        Don't just retrieve information from the retriever for the question; aim to engage the user and promote the product or service effectively. \
        If you don't know the answer, say that you don't know. \
        Make sure to summarize the answer where appropriate and construct responses that are easy to read and compelling for the user. \

        Sales Agent Role: As a sales agent, your goal is to highlight the benefits and unique selling points of the product or service. Tailor your responses to the user's persona and potential needs. Aim to build trust, create a sense of urgency, and guide the user toward making a positive decision about the product. Use positive language and emphasize the value the product can bring to the user's life. \

        When referring to video content, include the relevant timestamps. For example, "In the video at [00:02:15], it is explained that..."
        
        {context}
    """
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

def initialize_llm(api_key, model_name):
    """Initialize the LLM for summarizing the conversation."""
    llm = ChatGroq(model=model_name, groq_api_key=api_key)

    summary_system_prompt = """You are an assistant for summarizing conversations. 
    Given a conversation, generate a concise summary that captures the main points and key information. 
    If there is no conversation to summarize, state that there is no conversation to summarize.
    
    {context}
    """
    
    summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", summary_system_prompt),
            ("human", "{context}"),
        ]
    )
    
    summary_chain = create_stuff_documents_chain(llm, summary_prompt)
    
    return summary_chain

def process_data(docs_all):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len,
    )
    docs_chunks = text_splitter.split_documents(docs_all)
    
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    
    vectorstore = Chroma.from_documents(documents=docs_chunks, embedding=hf)
    return vectorstore

def load_data():
    web_loader = WebBaseLoader("https://www.apple.com/apple-vision-pro/")
    loader = YoutubeLoader.from_youtube_url(
        "https://www.youtube.com/watch?v=TX9qSaGXFyg", add_video_info=True
    )
    loader_pdf = PyPDFLoader("Apple_Vision_Pro_Privacy_Overview.pdf")
    loader_all = MergedDataLoader(loaders=[web_loader, loader, loader_pdf])
    return loader_all.load()

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
