# backend.py
# Flask + RAG backend with robust shims and optional local embedding fallback
# Replace your existing backend.py with this file.

import os
import tempfile
import traceback
import base64
import re
import asyncio
from collections import defaultdict

# ---------------------- Compatibility shim for older Python ----------------------
# Some google packages expect importlib.metadata.packages_distributions (added in Python 3.10).
# If it's missing (e.g. running on 3.9), provide a safe shim so imports don't fail.
try:
    import importlib.metadata as importlib_metadata
except Exception:
    # try backport package if available
    try:
        import importlib_metadata as importlib_metadata
    except Exception:
        importlib_metadata = None

if importlib_metadata is not None and not hasattr(importlib_metadata, "packages_distributions"):
    # Provide a very small shim that returns an empty mapping for requested package names.
    # This avoids AttributeError while not pretending to be a full implementation.
    def _packages_distributions(names):
        # names: iterable of package names
        # Return mapping pkg_name -> list of distributions that provide it.
        # We cannot reliably compute it on older Pythons without extra metadata; return empty lists.
        return {name: [] for name in names}
    importlib_metadata.packages_distributions = _packages_distributions
# --------------------------------------------------------------------------------

from flask import Flask, request, jsonify
from flask_cors import CORS

# PyMuPDF
import fitz  # pip install PyMuPDF

# LangChain and Google adapters
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Optional imports for local fallback embedding
_HAS_SENTENCE_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    _HAS_SENTENCE_TRANSFORMERS = False

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)  # Allow requests from the React frontend

# --- Global State (In-memory, for simplicity) ---
STATE = {
    "file_paths": {},
    "faiss_index_path": None,
    "temp_dir": None
}

# --- Helper Functions ---


def process_and_embed_pdfs(pdf_files, api_key, use_local_fallback=True):
    """
    Processes PDF files, extracts text, splits into chunks, creates a FAISS vector store.
    Primary embeddings: Google Generative Embeddings (via langchain_google_genai).
    If Google returns a quota error (or other embedding error) and a local fallback is available,
    fall back to sentence-transformers (all-MiniLM-L6-v2) to produce embeddings locally.
    Returns (True, message) on success, or (False, error_message) on failure.
    """
    if STATE.get("temp_dir") is None or not os.path.exists(STATE["temp_dir"]):
        STATE["temp_dir"] = tempfile.mkdtemp()

    file_paths = {}
    for file in pdf_files:
        temp_path = os.path.join(STATE["temp_dir"], file.filename)
        file.save(temp_path)
        file_paths[file.filename] = temp_path
    STATE["file_paths"] = file_paths

    docs = []
    for original_name, file_path in file_paths.items():
        try:
            doc_reader = fitz.open(file_path)
            for page_num, page in enumerate(doc_reader, 1):
                text = page.get_text()
                if text and text.strip():
                    docs.append(Document(page_content=text, metadata={'source': original_name, 'page': page_num}))
            doc_reader.close()
        except Exception as e:
            return False, f"Failed to open or read PDF {original_name}: {str(e)}"

    if not docs:
        return False, "No text could be extracted from uploaded PDFs. (Maybe scanned images without OCR?)"

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Try Google embeddings first
    try:
        # Create embeddings instance (will call Google Gemini embeddings)
        asyncio.set_event_loop(asyncio.new_event_loop())
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_documents(chunks, embedding=embeddings)
        STATE["faiss_index_path"] = os.path.join(STATE["temp_dir"], "faiss_index")
        vector_store.save_local(STATE["faiss_index_path"])
        return True, "Documents processed and indexed with Google embeddings successfully!"
    except Exception as google_exc:
        # Log the exception, then attempt optional local fallback
        msg = str(google_exc)
        app.logger.error("Google embedding error (will attempt fallback if enabled):\n" + traceback.format_exc())

        # Detect likely quota-related error messages so the frontend can show a helpful message
        if "quota" in msg.lower() or "rate-limit" in msg.lower() or "Quota exceeded" in msg:
            quota_msg = (
                "Google Generative Embeddings failed due to quota/rate limits. "
                "Possible causes: API key has no billing enabled, your project exceeded free-tier quotas, or rate limits were hit. "
                "Either enable billing / upgrade quota in the Google Cloud Console, use a different key, or use the optional local fallback (see logs)."
            )
        else:
            quota_msg = f"Google embedding failed: {msg}"

        # Optional local fallback using sentence-transformers
        if use_local_fallback and _HAS_SENTENCE_TRANSFORMERS:
            try:
                app.logger.info("Attempting local embedding fallback using sentence-transformers (all-MiniLM-L6-v2)...")
                model = SentenceTransformer("all-MiniLM-L6-v2")

                class SBertEmbeddings:
                    # Minimal wrapper exposing the embed_documents/embed_query that FAISS.from_documents expects.
                    def embed_documents(self, texts):
                        # sentence-transformers returns numpy arrays; convert to lists
                        embeds = model.encode(texts, show_progress_bar=False)
                        return [list(map(float, v)) for v in embeds]

                    def embed_query(self, text):
                        v = model.encode([text])[0]
                        return list(map(float, v))

                embeddings_local = SBertEmbeddings()
                vector_store = FAISS.from_documents(chunks, embedding=embeddings_local)
                STATE["faiss_index_path"] = os.path.join(STATE["temp_dir"], "faiss_index")
                vector_store.save_local(STATE["faiss_index_path"])
                return True, "Documents processed and indexed using local sentence-transformers embeddings (fallback)."
            except Exception as fallback_exc:
                app.logger.error("Local fallback embedding failed:\n" + traceback.format_exc())
                return False, f"{quota_msg} Also attempted local fallback but it failed: {str(fallback_exc)}"
        else:
            if use_local_fallback and not _HAS_SENTENCE_TRANSFORMERS:
                app.logger.warning("Local fallback requested but sentence-transformers is not installed.")
                return False, f"{quota_msg} Local fallback unavailable: sentence-transformers not installed."
            return False, quota_msg


def get_retrieval_filter_chain(api_key):
    """This AI chain selects the best source documents by returning their index numbers."""
    filter_prompt_template = """
    You are an expert research assistant. Your sole purpose is to analyze the numbered text passages below, which were retrieved for a user's question.
    Your must identify the passage numbers that are DIRECTLY and HIGHLY relevant to answering the user's question.

    - Review the User Question carefully.
    - Review each numbered Passage.
    - Your entire response MUST be a comma-separated list of the relevant passage numbers. Example: 0, 4, 9
    - Do NOT include any other text, headers, or explanations.
    - If and ONLY if none of the passages are relevant, your entire response MUST be the single word: None.

    User Question:
    "{question}"

    Retrieved Passages:
    {context}

    Your Response (only comma-separated numbers or "None"):
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0, google_api_key=api_key)
    prompt = PromptTemplate(template=filter_prompt_template, input_variables=["question", "context"])
    return prompt | model | StrOutputParser()


def get_answer_synthesis_chain(api_key):
    """This AI chain synthesizes the final answer from the filtered sources."""
    prompt_template = """You are a helpful assistant. Answer the question as detailed as possible based ONLY on the provided context. If the answer is not in the context, just say, "The answer is not available in the documents." Do not provide any information that is not in the context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    runnable = (
        {"context": lambda x: format_docs(x["input_documents"]), "question": lambda x: x["question"]}
        | prompt
        | model
        | StrOutputParser()
    )
    return runnable


def render_pdf_page_with_highlights(source_filename, page_num, texts_to_highlight):
    """
    Open the saved PDF and add highlight annotations for all exact text matches found.
    Returns PNG bytes or None on failure.
    """
    file_path = STATE["file_paths"].get(source_filename)
    if not file_path:
        return None
    try:
        doc = fitz.open(file_path)
        page = doc.load_page(page_num - 1)
        for text in texts_to_highlight:
            if not text or not text.strip():
                continue
            try:
                areas = page.search_for(text, hit_max=32)  # limit highlights per text
                for area in areas:
                    page.add_highlight_annot(area)
            except Exception:
                # search_for can raise for extremely long texts; ignore and continue
                continue
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes
    except Exception:
        app.logger.error("Failed to render highlighted page:\n" + traceback.format_exc())
        return None


# --- API Endpoints ---


@app.route("/api/process-pdfs", methods=["POST"])
def process_pdfs_endpoint():
    if "files" not in request.files:
        return jsonify({"error": "No files part"}), 400
    files = request.files.getlist("files")
    api_key = request.form.get("apiKey")
    if not api_key:
        return jsonify({"error": "API key is required"}), 400

    # allow client to opt-out of local fallback
    use_local_fallback = request.form.get("useLocalFallback", "true").lower() in ("1", "true", "yes")

    try:
        success, message = process_and_embed_pdfs(files, api_key, use_local_fallback=use_local_fallback)
        if success:
            return jsonify({"message": message})
        else:
            return jsonify({"error": message}), 400
    except Exception as e:
        app.logger.error(f"Error in /api/process-pdfs: {traceback.format_exc()}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/api/ask", methods=["POST"])
def ask_endpoint():
    data = request.json
    raw_question, api_key = data.get("question"), data.get("apiKey")
    if not all([raw_question, api_key, STATE.get("faiss_index_path")]):
        return jsonify({"error": "Missing data or documents not processed"}), 400

    # ========================== FIX: QUERY CLEANING ==========================
    possible_questions = [q.strip() for q in raw_question.strip().split("\n") if q.strip()]
    if possible_questions:
        actual_question = possible_questions[-1]
        app.logger.info(f"Extracted actual question: '{actual_question}' from full input.")
    else:
        actual_question = raw_question
    # ======================================================================

    # Truncate the final question if it's too long for the embedding model
    MAX_QUERY_CHARS = 8000
    if len(actual_question) > MAX_QUERY_CHARS:
        app.logger.warning(f"Query truncated from {len(actual_question)} to {MAX_QUERY_CHARS} characters.")
        actual_question = actual_question[:MAX_QUERY_CHARS]

    try:
        asyncio.set_event_loop(asyncio.new_event_loop())
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        db = FAISS.load_local(STATE["faiss_index_path"], embeddings, allow_dangerous_deserialization=True)

        candidate_docs = db.similarity_search(actual_question, k=10)

        if not candidate_docs:
            return jsonify({"answer": "I could not find any information related to your question in the documents.", "sources": []})

        context_string = "".join(f"--- Passage [{i}] ---\n{doc.page_content}\n\n" for i, doc in enumerate(candidate_docs))
        app.logger.info("--- CONTEXT FOR FILTER AI ---")
        app.logger.info(f"Question: {actual_question}")
        app.logger.info(context_string)
        app.logger.info("-----------------------------")

        filter_chain = get_retrieval_filter_chain(api_key)
        ai_response = filter_chain.invoke({"question": actual_question, "context": context_string})

        app.logger.info(f"--- AI FILTER RAW RESPONSE ---\n{ai_response}\n------------------------")

        final_docs = []
        try:
            if "none" not in ai_response.lower():
                relevant_indices_str = re.findall(r'\d+', ai_response)
                relevant_indices = [int(i) for i in relevant_indices_str]
                final_docs = [candidate_docs[i] for i in relevant_indices if i < len(candidate_docs)]

            if not final_docs:
                app.logger.warning("AI filter reviewed the passages and found no relevant sources.")
        except Exception as e:
            app.logger.error(f"Error parsing AI filter response: {e}.")

        if not final_docs:
            return jsonify({"answer": "The answer is not available in the documents.", "sources": []})

        answer_chain = get_answer_synthesis_chain(api_key)
        llm_response = answer_chain.invoke({"input_documents": final_docs, "question": actual_question})

        grouped_sources = defaultdict(list)
        for doc in final_docs:
            grouped_sources[(doc.metadata.get("source"), doc.metadata.get("page"))].append(doc.page_content)

        sources = []
        for (source, page), texts in grouped_sources.items():
            img_bytes = render_pdf_page_with_highlights(source, page, texts)
            if img_bytes:
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                sources.append({"source": source, "page": page, "base64Image": f"data:image/png;base64,{img_base64}"})

        return jsonify({"answer": llm_response, "sources": sorted(sources, key=lambda x: x["page"])})
    except Exception as e:
        app.logger.error(f"Error in /api/ask: {traceback.format_exc()}")
        # If embeddings or Google APIs are failing due to quota, forward a helpful message
        err_str = str(e)
        if "Quota exceeded" in err_str or "quota" in err_str.lower():
            return jsonify({"error": "Embedding/query failed due to API quota/rate limits. Check billing/quotas or try using a different API key."}), 503
        return jsonify({"error": f"An error occurred during AI processing: {str(e)}"}), 500


if __name__ == "__main__":
    # Note: For production use, run under a WSGI server (gunicorn / uvicorn / etc.) and set debug=False.
    app.run(host="0.0.0.0", port=5001, debug=True)
