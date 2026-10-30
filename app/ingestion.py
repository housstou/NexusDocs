import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from app.config import settings

def extract_metadata_with_llm(text: str) -> dict:
    """Utilise le LLM pour extraire l'entreprise et la date d'incident d'un post-mortem."""
    llm = ChatGoogleGenerativeAI(model=settings.LLM_MODEL, temperature=0)
    prompt = f"""
    Analyse ce début de rapport d'incident (post-mortem) et extrait l'entreprise concernée et la date de l'incident.
    Renvoie UNIQUEMENT un objet JSON valide avec les clés "company" et "incident_date". Si non trouvé, mets "unknown".
    Texte : {text[:1500]}
    """
    try:
        response = llm.invoke(prompt)
        content = response.content.replace('```json', '').replace('```', '').strip()
        return json.loads(content)
    except Exception as e:
        print(f"Erreur d'extraction LLM: {e}")
        return {"company": "unknown", "incident_date": "unknown"}

def hierarchical_chunking(file_path: str) -> list[Document]:
    """Divise en grands chunks (parents) puis petits chunks (enfants) pour le contexte."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    global_meta = extract_metadata_with_llm(pages[0].page_content) if pages else {}
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    chunks = []
    for page_num, page in enumerate(pages):
        child_docs = child_splitter.split_text(page.page_content)
        for chunk_text in child_docs:
            chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    "source": os.path.basename(file_path),
                    "page": page_num + 1,
                    "company": global_meta.get("company", "unknown").lower(),
                    "incident_date": global_meta.get("incident_date", "unknown")
                }
            ))
    return chunks

def calculate_chunk_ids(chunks: list[Document]):
    """Calcul d'IDs déterministes pour éviter la duplication."""
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
        
    return chunks

def add_to_chroma(chunks: list[Document], vector_store) -> int:
    """Ajout incrémental dans ChromaDB."""
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    existing_items = vector_store.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks) > 0:
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        vector_store.add_documents(new_chunks, ids=new_chunk_ids)
        return len(new_chunks)
    else:
        print("No new documents to add")
        return 0