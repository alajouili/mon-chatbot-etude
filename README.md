# 🤖 Chatbot de Révision de Cours avec Ala (RAG System)

Bienvenue dans **Mon Professeur IA**, une solution intelligente conçue pour transformer vos supports de cours statiques en tuteurs interactifs. Cette application utilise une architecture de pointe appelée **RAG (Retrieval-Augmented Generation)** pour fournir des réponses précises basées exclusivement sur vos documents PDF.

## 🚀 Lien de l'application en direct
Accédez à l'outil ici : 
👉 **[TESTER L'APPLICATION MAINTENANT](https://mon-chatbot-etude-aeygqnmzxt2nbnsztv7djg.streamlit.app/)**

---

## 🧠 Qu'est-ce que le RAG ?
Contrairement à un chatbot classique qui utilise uniquement ses connaissances générales, ce projet implémente un système de **Génération Augmentée par Récupération (RAG)** :
1. **Ingestion** : L'application lit votre fichier PDF.
2. **Segmentation (Chunking)** : Le texte est découpé en petits morceaux optimisés pour l'analyse.
3. **Vectorisation (Embeddings)** : Chaque morceau est transformé en coordonnées mathématiques (vecteurs) via ChromaDB.
4. **Récupération (Retrieval)** : Quand vous posez une question, l'IA cherche les morceaux les plus pertinents dans votre cours.
5. **Génération** : L'IA utilise le contexte trouvé pour répondre précisément via Llama 3.

---

## ✨ Fonctionnalités clés
* **Analyse de PDF Multi-pages** : Téléchargez vos cours complets directement dans l'interface.
* **Configuration Personnalisée** : Entrez votre propre clé API Groq pour une utilisation sécurisée.
* **Rapidité Extrême** : Propulsé par les modèles **Llama 3** via l'infrastructure **Groq**.
* **Interface Intuitive** : Design épuré et facile d'utilisation créé avec **Streamlit**.

---

## 🛠️ Stack Technique
* **Framework IA** : LangChain
* **Interface** : Streamlit
* **Modèle de Langue (LLM)** : Groq / Llama 3
* **Base de Données Vectorielle** : ChromaDB
* **Langage** : Python 3.12

---

## 💻 Installation et Lancement Local

### 1. Prérequis
Assurez-vous d'avoir **Python 3.12** installé.

### 2. Installation
```bash
git clone [https://github.com/votre-nom/mon-chatbot-etude.git](https://github.com/votre-nom/mon-chatbot-etude.git)
cd mon-chatbot-etude
pip install -r requirements.txt
