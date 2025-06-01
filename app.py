
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the context, say: "I couldn't find that information in the provided PDFs. Please try rephrasing your question or refer to the uploaded documents." Do not make up answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# ---------- Main UI App ----------
def main():
    st.set_page_config(page_title="Smart PDF Chat", page_icon=":books:", layout="wide")

    st.markdown("""
        <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f4f7fa;
        }
        .stButton>button {
            background-color: #0056d2;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #0041a8;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
            padding: 12px;
            border: 1px solid #d1d1d1;
        }
        .st-chat-message-user {
            background-color: #e3f2fd;
            color: #0d47a1;
            padding: 10px 15px;
            border-radius: 20px 20px 0px 20px;
            margin-left: auto;
            margin-bottom: 10px;
            max-width: 80%;
        }
        .st-chat-message-assistant {
            background-color: #e8f5e9;
            color: #1b5e20;
            padding: 10px 15px;
            border-radius: 20px 20px 20px 0px;
            margin-right: auto;
            margin-bottom: 10px;
            max-width: 80%;
        }
        .chat-container {
            max-height: 65vh;
            overflow-y: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("📚 Smart PDF Chat Interface")
    st.markdown("Ask questions based on uploaded PDFs using **Gemini AI**.")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'processing_done' not in st.session_state:
        st.session_state.processing_done = False

    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input("1. Google API Key", type="password")
        pdf_docs = st.file_uploader("2. Upload PDF Files", type="pdf", accept_multiple_files=True)

        if st.button("🚀 Process PDFs"):
            if api_key and pdf_docs:
                with st.spinner("Processing PDFs..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks, api_key)
                        st.session_state.processing_done = True
                        st.session_state.conversation_history = []
                        st.success("✅ PDFs processed successfully!")
                    except Exception as e:
                        st.error(f"Processing error: {e}")
                        st.session_state.processing_done = False
            else:
                st.warning("Upload PDFs and enter your API key.")

        if st.button("🧹 Reset"):
            st.session_state.conversation_history = []
            st.session_state.processing_done = False
            st.rerun()

    with st.container():
        for user_msg, bot_msg, timestamp, source in st.session_state.conversation_history:
            with st.chat_message("user"):
                st.markdown(user_msg)
            with st.chat_message("assistant"):
                st.markdown(bot_msg)

    user_question = st.chat_input("💬 Ask something from the PDFs...")

    if user_question:
        if not api_key:
            st.error("Please enter your Google API Key.")
        elif not st.session_state.processing_done:
            st.warning("Upload and process your PDFs first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
                    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                    docs = db.similarity_search(user_question)
                    chain = get_conversational_chain(api_key)
                    result = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                    response = result["output_text"]

                    current_pdf_names = ", ".join([pdf.name for pdf in pdf_docs]) if pdf_docs else "N/A"
                    st.session_state.conversation_history.append(
                        (user_question, response, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), current_pdf_names)
                    )
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.conversation_history:
        with st.expander("📥 Download Conversation History"):
            df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Timestamp", "PDF Sources"])
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown(f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv">📄 Download CSV</a>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
