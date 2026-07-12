import os
import faiss
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

CONFIG = {
    "MODEL_NAME": "qwen3:0.6b",
    "EMBED_MODEL_NAME": "all-MiniLM-L6-v2",
    "PDF_PATH": "uploaded.pdf",
    "INDEX_PATH": "index.faiss",
    "METADATA_PATH": "metadata.npy",
    "TOP_K": 3,
}

embedding_model = SentenceTransformer(CONFIG["EMBED_MODEL_NAME"])

def split_metadata(raw_pages, chunk_size = 200, overlap = 50):
    structured_chunks = []
    chunk_id_counter = 0
    step = chunk_size - overlap

    for page_num, text in raw_pages:
        words = text.split()
        if not words:
            continue
    
        for i in range(0, len(words), step):
            chunk_words = words[i:i+chunk_size]
            chunks_text = (" ".join(chunk_words))
            
            structured_chunks.append({
                "chunk_id": chunk_id_counter,
                "page": page_num,
                "text": chunks_text
            })
            chunk_id_counter +=1
            
            if i + chunk_size >= len(words):
                break

    return structured_chunks
def build_and_save_knowledge_base():
    print(f"Processing '{CONFIG['PDF_PATH']}' and generathing fresh embeddings...")

    reader = PdfReader(CONFIG["PDF_PATH"])
    raw_pages = []
    
    for idx, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            raw_pages.append((idx + 1, page_text))
    
    chunks_metadata = split_metadata(raw_pages)
    
    text_strings = [item["text"] for item in chunks_metadata]
    embeddings = np.array(embedding_model.encode(text_strings), dtype = np.float32)
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, CONFIG["INDEX_PATH"])
    np.save(METADATA_PATH, np.array(chunks_metadata, dtype = object))
    print("Knowlede base successfully saved locally!\n")
    return index, chunks_metadata

def load_knowledge_base():
    if os.path.exists(CONFIG["INDEX_PATH"]) and os.path.exists(CONFIG["METADATA_PATH"]) and os.path.exists(CONFIG["PDF_PATH"]):
        pdf_modification_time = os.path.getmtime(CONFIG["PDF_PATH"])
        index_created_time = os.path.getmtime(CONFIG["INDEX_PATH"])
        
        if pdf_modification_time < index_created_time:
            print("Loading existing vectors from disk...")
            index = faiss.read_index(CONFIG["INDEX_PATH"])
            chunks_metadata = np.load(CONFIG["METADATA_PATH"], allow_pickle=True).tolist()
            return index, chunks_metadata

    return build_and_save_knowledge_base()
    
def retrieve_context(query, index, chunks_metadata, score_threshold=0.45):
    query_embedding = np.array(embedding_model.encode([query]), dtype = np.float32)
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, CONFIG["TOP_K"])

    top_docs = []
    citations = set()
    
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and score > score_threshold:
            matching_item = chunks_metadata[idx]
            top_docs.append(matching_item["text"])
            citations.add(f"Page {matching_item['page']}")
    if not top_docs:
        return "No relevent context found.", "No source"
    
    context_str = "\n\n".join(top_docs)
    source_str = ", ".join(sorted(citations))
    return context_str, source_str

def ask_llm(query, context, chat_history):
    history_str = ""
    for role, msg in chat_history:
        history_str += f"{role}: {msg}\n\n"
    
    SYSTEM_PROMPT = """You are answering questions about the provided document.
    If the answer exists in Context or in Conversation,answer using Context.
    If it is not present,
    say: "I couldn't find this information in the document."Do not invent document contents."""
    
    prompt = f"""
    
    Conversation:
    {history_str}
        
    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = ollama.chat(
        model = CONFIG["MODEL_NAME"],
        messages=[{"role": "system", "content": SYSTEM_PROMPT},{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()    

def run_chat():
    index, chunks_metadata = load_knowledge_base()
    
    print(f"Connecting to local Ollama instance using '{CONFIG['MODEL_NAME']}'...\n")
    chat_history = []

    while True:
        try:
            query = input("User : ").strip()
            if query.lower() == "exit":
                print("Exiting chat session.")
                break
            if not query:
                continue
                
            context, sources = retrieve_context(query, index, chunks_metadata)
            reply = ask_llm(query, context, chat_history)
            
            print(f"Assistant : {reply}\n")
            print(f"[Sources: {sources}]\n")
            
            chat_history.append(("User", query))
            chat_history.append(("Assistant", reply))
            chat_history = chat_history[-10:]
        except Exception as e:
            print(f"An unexpected error occurred: {e}\n")

if __name__ == "__main__":
    run_chat()