import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from google.generativeai.types.generation_types import StopCandidateException
from PyPDF2.errors import PdfReadError
from docx import Document

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# def get_pdf_text(pdf_docs):
#     text=""
#     for pdf in pdf_docs:
#         pdf_reader= PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text+= page.extract_text()
#     return  text

def get_text_from_file(file):
  """Extracts text from uploaded file based on its format."""
  extension = os.path.splitext(file.name)[1].lower()
  if extension == ".pdf":
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
      text += page.extract_text()
    return text
  elif extension == ".docx":
    document = Document(file)
    text = ""
    for paragraph in document.paragraphs:
      text += paragraph.text
    return text
  elif extension == ".txt":
    return file.read().decode("utf-8")
  else:
    return None  # Handle unsupported file formats


def get_pdf_text(pdf_docs):
  """Extracts text from a list of PDF documents."""
  text = ""
  for pdf in pdf_docs:
    text += get_text_from_file(pdf)
  return text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    You are an expert market researcher with 20 years of experience. For the last 10 years you have been doing research in the middle east region to understand consumer sentiment. 
    You are a post graduate in psycology and have done your doctorate in emotions and culture. You  have an in depth understanding of consumer language culture and geography of the middle east. Please act as SixthFactor the consumer sentiment expert  bot and using the consumer comments we have captured please answer any questions with as much detail as possible.  
    Please be professional and courteous when answering these queries.
    \n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Response:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.2)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization= True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    # print(response)
    # st.write("Reply: ", response["output_text"])
    return response["output_text"]

def update_and_display_conversation(user_question, response):
    # Initialize the conversation history if not already done
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Append the new question-response pair
    st.session_state.conversation_history.append((user_question, response))
    
    # Display the conversation history
    for question, response in st.session_state.conversation_history:
        st.text(f"Q: {question}")
        st.text(f"A: {response}")
        st.markdown("---")  # Separator for readability


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with Expert💁")
    user_question = st.chat_input("Ask a Question from the PDF Files")

    if user_question:
        try:
            response = user_input(user_question)
            update_and_display_conversation(user_question, response)
        except StopCandidateException:
            st.warning("Please try asking your question in a different way.")
        except Exception as e:
            # Handle any exception by displaying an error message
                st.error("An error occurred. Please try again.")
        
            

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            try:
                with st.spinner("Processing..."):
                    # Assuming get_pdf_text, get_text_chunks, and get_vector_store are defined functions
                    raw_text = get_pdf_text(pdf_docs)  # Extract text from uploaded PDFs
                    text_chunks = get_text_chunks(raw_text)  # Break text into manageable chunks
                    get_vector_store(text_chunks)  # Process text chunks and store vectors
                    st.success("Done")
            except PdfReadError:
                st.error("An error occurred while reading the PDF. Please ensure the uploaded file is a valid PDF.")
            # with st.spinner("Processing..."):
            #     raw_text = get_pdf_text(pdf_docs)
            #     text_chunks = get_text_chunks(raw_text)
            #     get_vector_store(text_chunks)
            #     st.success("Done")

if __name__ == "__main__":
    main()
