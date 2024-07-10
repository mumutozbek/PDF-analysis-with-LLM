import streamlit as st
import os
import json
import logging
import tempfile
import shutil
from langchain_community.llms import Ollama
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

# Set page configuration at the beginning
st.set_page_config(page_title="PDF Analysis with Llama3", layout="wide")

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

def create_vector_db(file_upload) -> Chroma:
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
        loader = UnstructuredPDFLoader(path)
        data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    logger.info("Document split into chunks")

    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings, collection_name="myRAG")
    logger.info("Vector DB created")

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db, chunks  # Return both the vector DB and the document chunks

def llm_respond(text, llm):
    summary_query = f"Summarize the following text:\n\n{text}"
    analysis_query = f"Analyze the following text and provide insights:\n\n{text}"
    
    summary_response = ""
    for response in llm.stream(summary_query):
        summary_response += response + " "
    summary_response = summary_response.strip()
    
    analysis_response = ""
    for response in llm.stream(analysis_query):
        analysis_response += response + " "
    analysis_response = analysis_response.strip()
    
    return summary_response, analysis_response

def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    logger.info(f"Processing question: {question} using model: {selected_model}")
    llm = ChatOllama(model=selected_model, temperature=0)
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(vector_db.as_retriever(), llm, prompt=QUERY_PROMPT)

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Only provide the answer from the {context}, nothing else.
    Add snippets of the context you used to answer the question.
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response

def main():
    st.title("PDF Analysis with Llama3")
    
    st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;
        }
        .chat-box {
            background-color: #ffffff;
            border: 1px solid #d0d0d0;
            border-radius: 5px;
            padding: 10px;
            height: 400px;
            overflow-y: scroll;
        }
    </style>
    """, unsafe_allow_html=True)
    
    models_info = ollama.list()
    available_models = [model["name"] for model in models_info["models"]]
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    with col1:
        st.header("Upload and Process")
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        
        if uploaded_file is not None:
            if not os.path.exists('temp'):
                os.makedirs('temp')
            
            file_path = os.path.join("temp", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.write(f"File saved to {file_path}")
            
            st.session_state["vector_db"], chunks = create_vector_db(uploaded_file)
            st.success("PDF processed and vector database created.")
            
            transcription_text = " ".join([chunk.page_content for chunk in chunks])
            st.session_state["transcription_text"] = transcription_text

            llm = Ollama(model="llama3")
            english_summary, english_analysis = llm_respond(transcription_text, llm)
            
            st.session_state["english_summary"] = english_summary
            st.session_state["english_analysis"] = english_analysis

    with col2:
        st.header("Analysis")
        if 'english_summary' in st.session_state:
            st.write("## English Summary")
            st.write(st.session_state.english_summary)
            st.write("## English Analysis")
            st.text_area("English Analysis", st.session_state.english_analysis, height=200)

    with col3:
        if available_models:
            selected_model = st.selectbox("Pick a model available locally on your system ↓", available_models)

        st.header("Chat Box")
        chat_box = st.empty()
        if 'english_summary' in st.session_state:
            chat_box.markdown(f"""
            <div class="chat-box">
                <p><strong>English Summary:</strong> {st.session_state.english_summary}</p>
                <p><strong>English Analysis:</strong> {st.session_state.english_analysis}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if 'transcription_text' in st.session_state:
            user_input = st.text_input("Ask a question based on the transcription:")
            if st.button("Send"):
                with st.spinner("Generating response..."):
                    response = process_question(user_input, st.session_state["vector_db"], selected_model)
                    chat_box.markdown(f"""
                    <div class="chat-box">
                        <p><strong>You:</strong> {user_input}</p>
                        <p><strong>LLM:</strong> {response}</p>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
