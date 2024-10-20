import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from streamlit_star_rating import st_star_rating
# LLM and retriever to generate final response
from langchain_community.llms import LlamaCpp
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

# Initialize Pinecone
pc = Pinecone(api_key="9acaef2a-77c1-44f9-8bf7-01378bd86793")
index_name = 'professor-reviews'
if index_name not in pc.list_indexes().names():
    # Create a serverless spec with cloud and region
    spec = ServerlessSpec(cloud='aws', region='us-east-1')  # Adjust the cloud and region as needed
    pc.create_index(name=index_name, dimension=1536, metric='euclidean', spec=spec)
index = pc.Index(index_name)

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit app config
st.set_page_config(page_title="Rate My Professor", layout="centered")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Review", "Interact"])

# Review page
if page == "Review":
    st.write("## Welcome to Rate My Professor!")
    st.write("Here you can submit reviews about your professors.")
    with st.form(key='professor_review_form'):
        prof_name = st.text_input("Professor's Name", placeholder="Enter professor's name", key="prof_name")
        department = st.text_input("Professor's Department", placeholder="Enter professor's department", key="department")
        review = st.text_input("Review", placeholder="Write your review here", key="review")
        rating = st_star_rating("Rate:", maxValue=5, defaultValue=0, key="rating", dark_theme=True)

        # Submit button
        submit_button = st.form_submit_button("Submit Review")
    
    if submit_button:
        if prof_name and department and review:
            review += f"The rating of this professor is {rating} stars"
            embedding = embedding_model.encode(review).tolist()
           
            review_id = f"{prof_name}_{department}_{rating}"
            metadata={
                "professor_name":prof_name,
                "professor_department":department,
                "review_text":review,
                "rating":rating

            }
            index.upsert([(review_id, embedding,metadata)])
            st.success("Review submitted successfully!")
            # Refreshes the page after submit button is pressed
            st.markdown('<meta http-equiv="refresh" content="0;url=/" />', unsafe_allow_html=True)
        else:
            st.error("Please fill in all the fields")

# Interact page
elif page == "Interact":
    st.write("## Chatbot")
    st.write("Ask me anything about the professors!")

    # Generate result from query
    def generate_result(query):
        query_embedding = embedding_model.encode(query).tolist()
        result = index.query(vector=query_embedding, top_k=5,include_metadata=True)
        return result

    # Extract context from Pinecone result
    def extract_context(result):
        context = ""
        for match in result['matches']:
            # Check if 'metadata' exists before accessing it
            if 'metadata' in match:
                metadata = match['metadata']
                context += (
                    f"Professor: {metadata.get('professor_name', 'N/A')}, "
                    f"Department: {metadata.get('professor_department', 'N/A')}, "
                    f"Review: {metadata.get('review_text', 'N/A')}, "
                    f"Rating: {metadata.get('rating', 'N/A')} stars\n"
                )
            else:
                context += f"Review ID: {match['id']} does not have metadata.\n"
        return context



    #LOAD THE LLAMA MODEL.

    llm = LlamaCpp(
        model_path=r"C:\Users\Abina\ratemyprof\llama\llama-2-7b-chat.Q8_0.gguf",
        temperature=0.5,
        max_tokens=2048,
        top_p=1,
    )
    
    def generate_response(query, result):
        # Extract the context from the result
        context = extract_context(result)

      
        template = """
            {context}
            You are an assistant that will help students find suitable professors and answer their questions.
            </s>
            <|user|>
            {query}
            </s>
            <|assistant|>
        """
        
        # Creating the prompt using ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template(template)
        
        # Prepare the context and query as inputs for the prompt
        formatted_input = {
            "context": context,
            "query": query
        }
        
        # Create the RAG chain properly using `RunnableSequence` for a sequence of operations
        rag_chain = (
            prompt | llm | StrOutputParser()
        )

    
        response = rag_chain.invoke(formatted_input)
        
        return response



    #User input for chatbot interaction
    user_input = st.text_input("Input your query:")

    if user_input:
        result = generate_result(user_input)
        
    
        response = generate_response(user_input, result) 
        st.write(response) 
      
         
