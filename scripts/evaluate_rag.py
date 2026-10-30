import sys
from unittest.mock import MagicMock
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
sys.modules['langchain_community.llms.vertexai'] = MagicMock()
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_precision, 
    context_recall
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Import depuis notre application
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.pipeline import AdvancedRAGPipeline
from app.config import settings

def main():
    os.environ["GOOGLE_API_KEY"] = settings.GEMINI_API_KEY
    
    print(" Initialisation du RAG pour évaluation...")
    rag = AdvancedRAGPipeline()
    
    # 1. Définition du dataset de test (Golden Dataset - Google Cloud Incident)
    questions = [
        # Q1 : Teste la capacité à identifier la root cause technique (Context Precision / Faithfulness)
        "Quelle a été la cause principale (root cause) de cet incident réseau ?",
        
        # Q2 : Teste l'extraction de temporalité et de durées (Context Recall)
        "Combien de temps l'incident a-t-il duré au total et à quelle date a-t-il eu lieu ?",
        
        # Q3 : Teste l'extraction de métadonnées très spécifiques à un service (Answer Relevancy)
        "Quel a été l'impact spécifique de la panne sur le service Cloud SQL, et dans quelles régions ?",
        
        # Q4 : Teste la compréhension des mécanismes de monitoring/mitigation
        "En combien de temps l'erreur a-t-elle été initialement détectée et par quel moyen ?"
    ]
    
    ground_truths = [
        
        "La cause principale était le déploiement d'une nouvelle fonctionnalité d'infrastructure qui a déclenché un bug latent dans le code interne du répartiteur de charge réseau (internal network load balancer).",
        
        
        "L'incident a eu lieu le 12 novembre 2021 et a duré 1 heure et 44 minutes (de 00h30 à 02h14, heure du Pacifique).",
        
        
        "Les clients Cloud SQL ont subi des échecs lors des requêtes de création de sondes (probe creation request failures). L'impact s'est produit dans les régions europe-west1, europe-west4, et europe-west5.",
        
        
        "L'erreur a été détectée en seulement 4 minutes grâce aux systèmes de sécurité automatisés de Google, ce qui a permis de ralentir la propagation de la panne."
    ]
    
    answers = []
    contexts = []
    
    print(" Génération des réponses via le pipeline hybride...")
    for q in questions:
        # Appel de notre API en interne
        res = rag.query(q)
        answers.append(res["answer"])
        
        # Extraction du texte des chunks retournés pour Ragas
        ctx = [source["text"] for source in res["sources"]]
        contexts.append(ctx)
        
    # 2. Formatage pour HuggingFace Datasets (requis par Ragas)
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    # 3. Évaluation "LLM as a Judge"
    print(" Démarrage de l'évaluation Ragas...")
    eval_llm = ChatGoogleGenerativeAI(model=settings.LLM_MODEL, temperature=0)
    eval_embeddings = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL)
    
    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=eval_llm,
        embeddings=eval_embeddings
    )
    
    print("\n === Résultats de l'évaluation === ")
    print(result)
    
    # 4. Export des résultats
    os.makedirs("results", exist_ok=True)
    df = result.to_pandas()
    df.to_csv("results/ragas_evaluation.csv", index=False)
    print("Résultats exportés dans results/ragas_evaluation.csv")

if __name__ == "__main__":
    main()