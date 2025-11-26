import streamlit as st
import os
import fitz  # PyMuPDF
import tempfile
import asyncio
import time  # Added for rate limiting
from collections import defaultdict
import re
import shutil

# --- LangChain Imports (Modern LCEL Approach) ---
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="AI-Powered PDF Assistant",
    page_icon="🤖",
    layout="wide",
)

# --- 2. State Management Initialization ---
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = None
if "file_paths" not in st.session_state:
    st.session_state.file_paths = {}
if "selected_message_index" not in st.session_state:
    st.session_state.selected_message_index = None
if "current_source_page_index" not in st.session_state:
    st.session_state.current_source_page_index = 0
if "previous_selected_message_index" not in st.session_state:
    st.session_state.previous_selected_message_index = None

# --- 3. Backend Helper Functions ---

def process_and_embed_pdfs(pdf_docs, api_key):
    """Handles PDF processing with Rate Limiting to prevent 429 Errors."""
    # Create temp directory
    if not st.session_state.temp_dir or not os.path.exists(st.session_state.temp_dir):
        st.session_state.temp_dir = tempfile.mkdtemp()
    
    st.session_state.file_paths = {}
    
    # Save uploaded files
    for doc in pdf_docs:
        temp_path = os.path.join(st.session_state.temp_dir, doc.name)
        with open(temp_path, "wb") as f:
            f.write(doc.getbuffer())
        st.session_state.file_paths[doc.name] = temp_path

    docs = []
    progress_bar = st.progress(0.0, text="Reading documents...")
    
    # Extract text
    for i, (original_name, file_path) in enumerate(st.session_state.file_paths.items()):
        try:
            doc_reader = fitz.open(file_path)
            for page_num, page in enumerate(doc_reader, 1):
                text = page.get_text()
                if text:
                    docs.append(Document(page_content=text, metadata={'source': original_name, 'page': page_num}))
            doc_reader.close()
        except Exception as e:
            st.error(f"Error reading {original_name}: {e}")
            return False
        progress_bar.progress((i + 1) / len(st.session_state.file_paths), text=f"Processed {original_name}")

    if not docs:
        st.warning("No text extracted. Check if PDFs are images.")
        return False
        
    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Embed with Rate Limiting (Batching)
    progress_bar.progress(0.0, text="Embedding text (Slow mode to avoid quotas)...")
    
    try:
        # Using newer text-embedding-004 model
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            google_api_key=api_key
        )
        
        vector_store = None
        
        # Batch size of 10 chunks to respect free tier limits
        batch_size = 10
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            
            # Create or update vector store
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embedding=embeddings)
            else:
                vector_store.add_documents(batch)
            
            # Progress update
            progress = min((i + batch_size) / total_chunks, 1.0)
            progress_bar.progress(progress, text=f"Embedding: {int(progress*100)}% complete")
            
            # CRITICAL: Sleep to prevent 429 Quota Exceeded
            time.sleep(2) 

        vector_store.save_local("faiss_index")
        progress_bar.empty()
        return True

    except Exception as e:
        st.error(f"Embedding failed. Error: {e}")
        return False

def get_retrieval_filter_chain(api_key):
    """Selects best source documents using Gemini 2.0 Flash."""
    filter_prompt_template = """
    You are a research assistant. Analyze the numbered passages below.
    Identify passage numbers DIRECTLY relevant to the user's question.
    Respond ONLY with a comma-separated list of numbers (e.g., 0, 5). If none, say "None".

    Question: "{question}"

    Passages:
    {context}

    Relevant Numbers:
    """
    # Updated to Gemini 2.0 Flash
    model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.0, google_api_key=api_key)
    prompt = PromptTemplate(template=filter_prompt_template, input_variables=["question", "context"])
    return prompt | model | StrOutputParser()

def get_answer_synthesis_chain(api_key):
    """Synthesizes answer using Gemini 2.0 Flash."""
    prompt_template = """
    You are a helpful assistant. Answer the question based ONLY on the provided context.
    If the answer is not in the context, say "The answer is not available in the documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    # Updated to Gemini 2.0 Flash
    model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt | model | StrOutputParser()

def render_pdf_page_with_highlights(source_filename, page_num, texts_to_highlight):
    """Renders PDF page with highlights."""
    file_path = st.session_state.file_paths.get(source_filename)
    if not file_path or not os.path.exists(file_path): 
        return None
        
    try:
        doc = fitz.open(file_path)
        if page_num < 1 or page_num > len(doc): return None
            
        page = doc.load_page(page_num - 1)
        for text in texts_to_highlight:
            clean_text = text.strip()
            if len(clean_text) > 5:
                areas = page.search_for(clean_text)
                for area in areas:
                    highlight = page.add_highlight_annot(area)
                    highlight.update()
                    
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes
    except Exception:
        return None

# --- 4. Main UI Application ---
def main():
    st.title("🤖 AI-Powered PDF Assistant")
    st.markdown("Powered by **Gemini 2.0 Flash** & **Text-Embedding-004**")

    with st.sidebar:
        st.header("⚙️ Setup")
        api_key = st.text_input("1. Google API Key", type="password", help="Required for Gemini AI.")
        pdf_docs = st.file_uploader("2. Upload PDF Files", type="pdf", accept_multiple_files=True)

        if st.button("🚀 Process Documents"):
            if api_key and pdf_docs:
                with st.spinner("Processing... this may take a moment to respect API limits."):
                    if process_and_embed_pdfs(pdf_docs, api_key):
                        st.session_state.processing_complete = True
                        st.session_state.conversation_history = []
                        st.session_state.selected_message_index = None
                        st.session_state.current_source_page_index = 0
                        st.success("✅ Documents processed!")
                        st.rerun()
            else:
                st.warning("Please provide API Key and PDFs.")
        
        if st.button("🧹 Reset App"):
            keys = ["conversation_history", "processing_complete", "file_paths", 
                    "selected_message_index", "current_source_page_index", "previous_selected_message_index"]
            for k in keys:
                if k in st.session_state: del st.session_state[k]
            
            if st.session_state.get("temp_dir") and os.path.exists(st.session_state.temp_dir):
                shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index", ignore_errors=True)
            st.rerun()

    if st.session_state.processing_complete:
        col1, col2 = st.columns([2, 1])

        def set_selected_message(index):
            st.session_state.selected_message_index = index
            st.session_state.current_source_page_index = 0

        with col1:
            st.subheader("💬 Chat")
            for i, turn in enumerate(st.session_state.conversation_history):
                with st.chat_message("user"): st.write(turn["user"])
                with st.chat_message("assistant"):
                    st.write(turn["assistant"])
                    st.button("Show Sources", key=f"btn_{i}", on_click=set_selected_message, args=(i,))

            user_question = st.chat_input("Ask a question about your documents...")
            if user_question:
                with st.chat_message("user"): st.write(user_question)

                with st.spinner("Thinking..."):
                    try:
                        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
                        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                        candidate_docs = db.similarity_search(user_question, k=15)
                        
                        context_str = ""
                        for i, doc in enumerate(candidate_docs):
                            context_str += f"--- Passage [{i}] ---\n{doc.page_content}\n\n"

                        # Step 1: Filter
                        filter_chain = get_retrieval_filter_chain(api_key)
                        ai_response = filter_chain.invoke({"question": user_question, "context": context_str})
                        
                        final_docs = []
                        try:
                            if "none" not in ai_response.lower():
                                indices = [int(i) for i in re.findall(r'\d+', ai_response)]
                                final_docs = [candidate_docs[i] for i in indices if i < len(candidate_docs)]
                            if not final_docs and candidate_docs: final_docs = candidate_docs[:5]
                        except: final_docs = candidate_docs[:5]

                        # Step 2: Answer
                        if not final_docs:
                            llm_response = "The answer is not available in the documents."
                        else:
                            answer_chain = get_answer_synthesis_chain(api_key)
                            combined_context = "\n\n".join([doc.page_content for doc in final_docs])
                            llm_response = answer_chain.invoke({"context": combined_context, "question": user_question})

                        st.session_state.conversation_history.append({"user": user_question, "assistant": llm_response, "sources": final_docs})
                        st.session_state.selected_message_index = len(st.session_state.conversation_history) - 1
                        st.session_state.current_source_page_index = 0
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error: {e}")

        with col2:
            st.subheader("🧐 References")
            if st.session_state.selected_message_index is not None:
                # Same pagination logic as before
                if st.session_state.selected_message_index != st.session_state.get('previous_selected_message_index'):
                    st.session_state.current_source_page_index = 0
                    st.session_state.previous_selected_message_index = st.session_state.selected_message_index

                selected_turn = st.session_state.conversation_history[st.session_state.selected_message_index]
                retrieved_sources = selected_turn.get("sources", [])
                
                if retrieved_sources:
                    grouped = defaultdict(list)
                    for doc in retrieved_sources:
                        grouped[(doc.metadata.get('source'), doc.metadata.get('page'))].append(doc.page_content)
                    
                    src_pages = sorted(grouped.items())
                    if src_pages:
                        if st.session_state.current_source_page_index >= len(src_pages):
                            st.session_state.current_source_page_index = 0
                        
                        (src, pg), texts = src_pages[st.session_state.current_source_page_index]
                        st.markdown(f"**{src} | Page {pg}**")
                        img = render_pdf_page_with_highlights(src, pg, texts)
                        if img: st.image(img, use_container_width=True)
                        else: st.warning("Preview unavailable.")
                        
                        st.markdown("---")
                        c1, c2, c3 = st.columns([1,2,1])
                        with c1: 
                            if st.button("⬅️", disabled=st.session_state.current_source_page_index<=0):
                                st.session_state.current_source_page_index -= 1
                                st.rerun()
                        with c2: st.markdown(f"<center>{st.session_state.current_source_page_index+1}/{len(src_pages)}</center>", unsafe_allow_html=True)
                        with c3:
                            if st.button("➡️", disabled=st.session_state.current_source_page_index>=len(src_pages)-1):
                                st.session_state.current_source_page_index += 1
                                st.rerun()
                else: st.info("No sources cited.")
            else: st.info("Select 'Show Sources' on a message.")
    else:
        st.info("Upload documents to begin.")

if __name__ == "__main__":
    main()