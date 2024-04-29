# Local RAG System with Ollama

## Project Description

This project aims to build a local Retrieval-Augmented Generation (RAG) system using Ollama, a language model developed by the Langchain community. The system is designed to answer questions based on contextual information stored in a local database. Below is a brief overview of each Python file in the project:

1. **get_embedding_function.py:**
   - This script defines a function `get_embedding_function()` that returns an embedding function used for encoding text into numerical representations. It imports embedding modules from `langchain_community` and provides flexibility to choose between different models. Users are instructed to change the model parameter as per their requirements or downloaded models.

2. **populate_database.py:**
   - This script populates a local database with documents stored in a specified directory. It utilizes functions to load PDF documents, split them into chunks of text, calculate chunk IDs, and add them to a vector storage (Chroma) using the embedding function obtained from `get_embedding_function.py`. Users can reset the database using the `--reset` flag.

3. **query_data.py:**
   - This script handles user queries by searching the populated database for relevant information. It uses the Chroma vector store and the Ollama language model to perform similarity search and generate responses based on the retrieved context. Users can input their queries via the command line interface.

4. **requirements.txt:**
   - This file lists the required Python packages for running the project, including dependencies like PyPDF, langchain, chromadb, and pytest.

5. **test_rag.py:**
   - This script contains unit tests to validate the accuracy of the RAG system's responses. It includes functions to test specific questions against expected responses, utilizing the `query_rag()` function from `query_data.py`. Tests cover various scenarios related to different board game rules.

6. **data folder:**
   - This directory stores the PDF documents from which the RAG system extracts contextual information.

