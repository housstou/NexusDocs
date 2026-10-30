# NexusDocs 

**Enterprise-Grade Advanced RAG Architecture for Engineering Post-Mortems**

NexusDocs est un moteur de recherche et de question-réponse (RAG) ultra-optimisé, conçu pour analyser des documents techniques complexes (notamment des rapports d'incidents et post-mortems). 

Cette V2 introduit une architecture robuste intégrant l'ingestion incrémentale, le pre-filtering par LLM, la fusion RRF, le reranking et le cache sémantique.

##  Architecture Core Features

### 1. Smart Ingestion & Metadata Extraction
* **Hierarchical Chunking** : Découpage intelligent.
* **LLM Metadata Pre-filtering** : Lors de l'ingestion, un LLM extrait automatiquement les entités clés (`company`, `incident_date`) de la première page. Ces données alimentent les métadonnées ChromaDB pour permettre un filtrage exact (`where` clause).
* **Incremental Updates** : Génération d'IDs déterministes et séquentiels (`source:page:chunk`). Le système détecte les IDs existants et n'indexe que les différentiels.

### 2. Advanced Retrieval Pipeline
* **Semantic Caching (Redis)** : Mise en cache sémantique des requêtes et réponses via `RedisSemanticCache` pour réduire drastiquement la latence sur les questions similaires.
* **Query Expansion** : Génération de reformulations de la requête utilisateur (sans approche HyDE pour éviter l'hallucination) afin de maximiser le *Recall*.
* **Hybrid Search (Dense + Sparse)** :
  * *Dense Retrieval* : ChromaDB + Google Generative AI Embeddings.
  * *Sparse Retrieval* : BM25 pour les acronymes et termes techniques précis.
* **Reciprocal Rank Fusion (RRF)** : Algorithme fusionnant les résultats du Dense et Sparse Search.

### 3. Re-ranking & Generation
* **Cross-Encoder Re-ranking** : Utilisation de `ms-marco-MiniLM-L-6-v2` pour scorer les paires issues du RRF.
* **Grounded Generation** : LLM Gemini 3.5 Flash Lite avec prompt strict anti-hallucination.

##  Tech Stack
* **Backend** : FastAPI (Asynchrone), Python 3.11
* **Orchestration RAG** : LangChain
* **Vector Store** : ChromaDB (Persistent)
* **Caching** : Redis 7
* **Embeddings & LLM** : Google Gemini API
* **Infrastructure** : Docker & Docker Compose

##  LLMOps & Évaluation (RAGAS)
Pour garantir l'absence d'hallucinations et la pertinence du retrieval, le système est évalué objectivement avec le framework **Ragas** (LLM-as-a-Judge) sur 4 métriques clés :
- **Faithfulness** : Vérifie que la réponse générée ne contient aucune information inventée par rapport aux chunks récupérés.
- **Answer Relevancy** : Évalue si la réponse répond directement à la question sans hors-sujet.
- **Context Precision** : Mesure si le *Cross-Encoder* a bien placé les chunks les plus pertinents tout en haut du contexte.
- **Context Recall** : Vérifie que notre recherche hybride (BM25 + Dense) a bien récupéré toutes les informations nécessaires à la réponse.

Un script d'évaluation est disponible dans `scripts/evaluate_rag.py`.

##  Quickstart
```bash
git clone https://github.com/housstou/NexusDocs.git
cd NexusDocs
cp .env.example .env
# Ajouter GEMINI_API_KEY dans le .env
docker compose up --build -d
