# Import necessary libraries and modules
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai._common import GoogleGenerativeAIError
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file and configure Google API
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    st.error("Google API key not found. Please check your environment variables.")
    st.stop()

# Function to extract text from a list of uploaded PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the extracted text into smaller chunks for processing
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from the text chunks using embeddings
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")
def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True  # Operation was successful
    except GoogleGenerativeAIError as e:
        print(f"Error embedding content we will get back to you: {e}")  # Log the error for debugging
        return False  # Operation failed


# Function to initialize and return a conversational chain for processing and answering questions
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Initialize conversation history in session state if not already present
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Function to handle user input, process it using the conversational chain, and update conversation history
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Preparing context from conversation history and current question for the model
    conversation_context = '\n'.join(["Q: " + q + "\nA: " + a for q, a in st.session_state.conversation_history])
    current_context = f"Q: {user_question}\nA: "
    full_context = conversation_context + current_context if conversation_context else current_context

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": full_context}, return_only_outputs=True)
    
    # Update the conversation history in the session state
    st.session_state.conversation_history.append((user_question, response["output_text"]))
    # st.write("Reply: ", response["output_text"])

# Function to display the conversation history
def display_history():
    for i, (q, a) in enumerate(st.session_state.conversation_history, 1):
        st.text(f"Q{i}: {q}\nA{i}: {a}\n")


# Main function to configure the app and handle user interactions
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with Model💁")

    # Initialize conversation history if not already initialized
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # User input for asking questions
    # user_question = st.text_input("Ask a Question from the PDF Files")
    user_question  = st.chat_input("Ask a question")
    if user_question:
        user_input(user_question)
        display_history()  # Display the conversation history
        
    # Sidebar for uploading PDF files and processing them
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        # Continuation from the previous code block
        if st.button("Submit & Process"):
            # Display a spinner while processing the PDF files
            with st.spinner("Processing..."):
                # Extract text from the uploaded PDF files
                raw_text = get_pdf_text(pdf_docs)
                # Split the extracted text into chunks for processing
                text_chunks = get_text_chunks(raw_text)
                # Create a vector store from the text chunks using embeddings
                get_vector_store(text_chunks)
                # Notify the user that processing is complete
                st.success("Processing complete!")
                # Optionally, you can display the raw text or chunks to give feedback to the user
                # st.write(raw_text) # Uncomment to display the raw extracted text

# Entry point for the Streamlit application
if __name__ == "__main__":
    main()
