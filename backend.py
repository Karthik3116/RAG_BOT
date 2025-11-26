import os
import tempfile
import traceback
import base64
import re
import asyncio
import time
from collections import defaultdict

# ---------------------- Compatibility shim for older Python ----------------------
try:
    import importlib.metadata as importlib_metadata
except Exception:
    try:
        import importlib_metadata as importlib_metadata
    except Exception:
        importlib_metadata = None

if importlib_metadata is not None and not hasattr(importlib_metadata, "packages_distributions"):
    def _packages_distributions(names):
        return {name: [] for name in names}
    importlib_metadata.packages_distributions = _packages_distributions
# --------------------------------------------------------------------------------

from flask import Flask, request, jsonify
from flask_cors import CORS

# PyMuPDF
import fitz

# LangChain Imports (Modern LCEL Approach)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# --- Global State ---
# Note: In a production environment, use a database (Redis/Postgres) instead of global variables.
STATE = {
    "file_paths": {},
    "faiss_index_path": None,
    "temp_dir": None
}

# --- Backend Helper Functions ---

def process_and_embed_pdfs(pdf_files, api_key):
    """
    Handles PDF processing with Rate Limiting to prevent 429 Errors.
    Matches the logic of the latest app.py.
    """
    # 1. Setup Temp Directory
    if STATE.get("temp_dir") is None or not os.path.exists(STATE["temp_dir"]):
        STATE["temp_dir"] = tempfile.mkdtemp()

    # 2. Save Files
    file_paths = {}
    for file in pdf_files:
        temp_path = os.path.join(STATE["temp_dir"], file.filename)
        file.save(temp_path)
        file_paths[file.filename] = temp_path
    STATE["file_paths"] = file_paths

    docs = []
    
    # 3. Extract Text
    for original_name, file_path in file_paths.items():
        try:
            doc_reader = fitz.open(file_path)
            for page_num, page in enumerate(doc_reader, 1):
                text = page.get_text()
                if text and text.strip():
                    docs.append(Document(page_content=text, metadata={'source': original_name, 'page': page_num}))
            doc_reader.close()
        except Exception as e:
            return False, f"Failed to read PDF {original_name}: {str(e)}"

    if not docs:
        return False, "No text extracted. Please check if your PDFs are scanned images."

    # 4. Split Text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # 5. Embed with Rate Limiting (Batching)
    try:
        # Asyncio loop fix for some environments
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        # UPDATED: Use text-embedding-004
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            google_api_key=api_key
        )
        
        vector_store = None
        
        # Batch size of 10 chunks to respect free tier limits
        batch_size = 10
        total_chunks = len(chunks)
        
        app.logger.info(f"Starting embedding for {total_chunks} chunks...")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            
            # Create or update vector store
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embedding=embeddings)
            else:
                vector_store.add_documents(batch)
            
            # CRITICAL: Sleep to prevent 429 Quota Exceeded (Same as app.py)
            time.sleep(2) 

        STATE["faiss_index_path"] = os.path.join(STATE["temp_dir"], "faiss_index")
        vector_store.save_local(STATE["faiss_index_path"])
        return True, "Documents processed successfully with Rate Limiting!"

    except Exception as e:
        app.logger.error(traceback.format_exc())
        return False, f"Embedding failed: {str(e)}"


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
    # UPDATED: Gemini 2.0 Flash
    model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.0, google_api_key=api_key)
    prompt = PromptTemplate(template=filter_prompt_template, input_variables=["question", "context"])
    # Modern LCEL
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
    # UPDATED: Gemini 2.0 Flash
    model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Modern LCEL
    return prompt | model | StrOutputParser()


def render_pdf_page_with_highlights(source_filename, page_num, texts_to_highlight):
    """Renders PDF page with highlights as bytes."""
    file_path = STATE["file_paths"].get(source_filename)
    if not file_path: return None
        
    try:
        doc = fitz.open(file_path)
        # Validate page
        if page_num < 1 or page_num > len(doc): return None

        page = doc.load_page(page_num - 1)
        for text in texts_to_highlight:
            clean_text = text.strip()
            if len(clean_text) > 5:
                # search_for can raise errors on weird pdf text, suppress
                try:
                    areas = page.search_for(clean_text, hit_max=10)
                    for area in areas:
                        page.add_highlight_annot(area)
                except Exception:
                    continue
                    
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes
    except Exception:
        app.logger.error(traceback.format_exc())
        return None


# --- API Endpoints ---

@app.route("/api/process-pdfs", methods=["POST"])
def process_pdfs_endpoint():
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    
    files = request.files.getlist("files")
    api_key = request.form.get("apiKey")
    
    if not api_key:
        return jsonify({"error": "API key is required"}), 400

    try:
        success, message = process_and_embed_pdfs(files, api_key)
        if success:
            return jsonify({"message": message})
        else:
            return jsonify({"error": message}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/api/ask", methods=["POST"])
def ask_endpoint():
    data = request.json
    user_question = data.get("question")
    api_key = data.get("apiKey")

    if not all([user_question, api_key, STATE.get("faiss_index_path")]):
        return jsonify({"error": "Missing data or documents not processed"}), 400

    try:
        # Asyncio loop fix
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        # UPDATED: Use text-embedding-004 for retrieval
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        db = FAISS.load_local(STATE["faiss_index_path"], embeddings, allow_dangerous_deserialization=True)

        # 1. Retrieval
        candidate_docs = db.similarity_search(user_question, k=15) # Matches app.py k=15

        if not candidate_docs:
             return jsonify({"answer": "I could not find any information related to your question in the documents.", "sources": []})

        context_str = ""
        for i, doc in enumerate(candidate_docs):
            context_str += f"--- Passage [{i}] ---\n{doc.page_content}\n\n"

        # 2. Filter Step
        filter_chain = get_retrieval_filter_chain(api_key)
        ai_response = filter_chain.invoke({"question": user_question, "context": context_str})

        final_docs = []
        try:
            if "none" not in ai_response.lower():
                indices = [int(i) for i in re.findall(r'\d+', ai_response)]
                final_docs = [candidate_docs[i] for i in indices if i < len(candidate_docs)]
            
            # Fallback Logic from app.py
            if not final_docs and candidate_docs:
                final_docs = candidate_docs[:5]
        except:
            final_docs = candidate_docs[:5]

        if not final_docs:
             return jsonify({"answer": "The answer is not available in the documents.", "sources": []})

        # 3. Answer Step
        answer_chain = get_answer_synthesis_chain(api_key)
        combined_context = "\n\n".join([doc.page_content for doc in final_docs])
        llm_response = answer_chain.invoke({"context": combined_context, "question": user_question})

        # 4. Process Visual Sources
        grouped_sources = defaultdict(list)
        for doc in final_docs:
            grouped_sources[(doc.metadata.get("source"), doc.metadata.get("page"))].append(doc.page_content)

        sources = []
        for (source, page), texts in grouped_sources.items():
            img_bytes = render_pdf_page_with_highlights(source, page, texts)
            if img_bytes:
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                sources.append({
                    "source": source, 
                    "page": page, 
                    "base64Image": f"data:image/png;base64,{img_base64}"
                })

        return jsonify({
            "answer": llm_response, 
            "sources": sorted(sources, key=lambda x: x["page"])
        })

    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)