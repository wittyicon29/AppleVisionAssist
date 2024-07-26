# **Project Overview**: 

The goal of the AppleVisionAssist project is to develop a sophisticated conversational AI chatbot that serves both as an information resource and a sales agent for Apple's Vision Pro product. The chatbot aims to provide users with detailed product insights, answer queries, and effectively promote the features and benefits of the Vision Pro.

**Technical Stack:**
- **Language Model:** Powered by advanced LLMs (Large Language Models) like those available from OpenAI or custom models using Hugging Face transformers.
- **Document Handling:** Integration with document loaders and retrievers for sourcing relevant information from diverse data sets.
- **Embeddings and Vector Search:** Utilize tools like Chroma and HuggingFace embeddings for efficient information retrieval.
- **Frontend Interface:** Built using user-friendly frameworks, ensuring a seamless user experience.

# **Data Sources**
- [Apple Vision Pro Privacy Overview](https://www.apple.com/privacy/docs/Apple_Vision_Pro_Privacy_Overview.pdf)
- [Official landing Page](https://www.apple.com/apple-vision-pro/)
- [Apple Vision Pro Youtube](https://www.youtube.com/watch?v=TX9qSaGXFyg)

# **Structure**
- data_loading.py: Contains functions to load data from the web, PDF and youtube video.
- processing.py: Functions to split text into chunks and generate embeddings.
- model_initialization.py: Code to initialize the model and retrieval chain.
- main.py: Streamlit application for the chatbot interface.

# Steps 
![PDF](https://github.com/wittyicon29/QABot-with-Conversational-Memory/assets/99320225/5832d0be-a092-4acb-97c0-d7fc7657942b)

#### Loading Data

1. **WebBaseLoader**: Fetches and loads web pages.
2. **PyPDFLoader**: Loads and parses the PDF containing milestone papers.
3. **MergedDataLoader**: Merges the data from the web and PDF loaders.

#### Processing Data

1. **Text Splitting**: 
    - `RecursiveCharacterTextSplitter` divides the loaded text into smaller, overlapping chunks to ensure that context is preserved.
    
2. **Embedding Generation**:
    - `HuggingFaceBgeEmbeddings` generates embeddings for the text chunks using a pre-trained model.
    
3. **Vector Store**:
    - The Chroma vector store is used to store and index these embeddings, enabling efficient retrieval.

#### Initializing the Model

1. **LLM Initialization**:
    - `ChatGroq` initializes the chosen LLM model using the provided API key.
    - 
2. **Prompt Templates**:
    - Custom prompt templates are created to reformulate user queries and generate responses based on the retrieved context.
    
3. **Retrieval Chain**:
    - A retrieval chain is created that uses a history-aware retriever to provide context-aware answers.

### Application

A Streamlit application allows users to interact with the chatbot. Key features include:
- **Input Query**: Users can enter natural language queries.
- **Chat History**: The system maintains context across multiple queries.
- **Display of Sources**: The sources used to generate answers are displayed, ensuring transparency.

### Improvements and Future Work

- **Citation and Reference Handling**: To develop a more robust system for citing YouTube videos, specifically focusing on accurate timestamp integration. This will ensure that references are clear and allow users to easily locate pertinent segments of video content.

### Setup Instructions

1. **Clone the Repository**:
    ```sh
    git clone <repository-url>
    cd <repository-folder>
    ```

2. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the Application**:
    ```sh
    streamlit run src/app.py
    ```

[Demo](https://github.com/user-attachments/assets/85868705-cc52-481a-b784-1b63d1b422ee)


