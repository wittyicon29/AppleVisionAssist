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
