import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Mon Professeur IA", page_icon="🤖")

st.title("🤖 Chatbot de Révision de Cours avec Ala")
st.write("Télécharge ton cours en PDF et pose tes questions !")

# --- SIDEBAR (Barre latérale pour la clé et le fichier) ---
with st.sidebar:
    st.header("1. Configuration")
    api_key = st.text_input("Entre ta clé Groq API ici :", type="password")
    st.markdown("[Clique ici pour avoir une clé gratuite](https://console.groq.com/keys)")
    
    st.header("2. Ton Cours")
    uploaded_file = st.file_uploader("Dépose ton PDF ici", type="pdf")

# --- FONCTIONNEMENT PRINCIPAL ---
if uploaded_file is not None and api_key:
    os.environ["GROQ_API_KEY"] = api_key
    
    # Message d'attente
    with st.spinner("Analyse du document en cours..."):
        try:
            # Sauvegarde temporaire du fichier pour le lire
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Chargement et découpage
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # Création de la mémoire (VectorStore)
            embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function)
            retriever = vectorstore.as_retriever()
            
            # Configuration du LLM
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
            
            st.success("✅ Analyse terminée ! Pose ta question ci-dessous.")
            
            # Zone de Chat
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if question := st.chat_input("Pose ta question sur le cours..."):
                # Afficher la question de l'utilisateur
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                # Réponse de l'IA
                with st.chat_message("assistant"):
                    relevant_docs = retriever.invoke(question)
                    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
                    prompt = f"Tu es un professeur expert. Réponds en te basant UNIQUEMENT sur ce contexte : {context_text}\n\nQuestion : {question}"
                    
                    response = llm.invoke(prompt)
                    st.markdown(response.content)
                    
                st.session_state.messages.append({"role": "assistant", "content": response.content})

        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")

elif uploaded_file is None:
    st.info("👈 Commence par déposer ton fichier PDF dans le menu à gauche.")
elif not api_key:
    st.warning("👈 Entre ta clé API Groq dans le menu à gauche pour démarrer.")
