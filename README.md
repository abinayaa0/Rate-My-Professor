# **Rate My Professor with LLM Integration**

This is a Streamlit-based application designed for submitting and interacting with professor reviews. It features a two-page interface where users can submit reviews, rate professors, and interact with a chatbot that retrieves reviews and generates responses based on user queries using a Large Language Model (LLM).

## **Features**
- **Submit Reviews:** Users can input professor names, departments, reviews, and ratings.
- **Star Rating System:** Provides a 5-star rating system for each professor.
- **Embedding-based Search:** Reviews are encoded using `SentenceTransformer` embeddings, stored, and queried using Pinecone’s vector database.
- **LLM Interaction:** A chatbot interface is powered by the Llama 2 model to provide meaningful responses based on professor review data.
  
## **Tech Stack**
- **Streamlit**: For building the web interface.
- **Pinecone**: Used as a vector database for efficient storage and querying of professor reviews.
- **SentenceTransformer**: `all-MiniLM-L6-v2` is used to encode reviews into vector representations.
- **LlamaCpp**: Llama 2 model is used to process the context and generate responses.
- **Langchain**: To integrate LLMs and design the prompt templates.

## **Setup Instructions**
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
