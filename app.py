import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configuration de la page
st.set_page_config(page_title="Mon Assistant RAG", page_icon="🤖")
st.title("🤖 Assistant d'Étude (Via Groq)")

# --- 1. Récupération de la clé API ---
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("Erreur : La clé API Groq est manquante dans les secrets.")
    st.stop()

# --- 2. Initialisation de la Session ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- 3. Barre latérale pour le fichier ---
with st.sidebar:
    st.header("📁 Tes Cours")
    uploaded_file = st.file_uploader("Dépose ton PDF ici", type="pdf")
    
    if uploaded_file and st.session_state.vectorstore is None:
        with st.spinner("Analyse du document en cours..."):
            # Sauvegarde temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Lecture
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            # Découpage
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            
            # Indexation
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            
            # Sauvegarde dans la mémoire
            st.session_state.vectorstore = vectorstore
            os.remove(tmp_path)
            st.success("Document prêt !")

# --- 4. Interface de Chat ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_text := st.chat_input("Pose ta question sur le cours..."):
    st.chat_message("user").markdown(prompt_text)
    st.session_state.messages.append({"role": "user", "content": prompt_text})

    if st.session_state.vectorstore is not None:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
        
        retriever = st.session_state.vectorstore.as_retriever()
        
        system_prompt = (
            "Tu es un assistant pédagogique. Utilise le contexte ci-dessous pour répondre. "
            "Si tu ne sais pas, dis-le. Sois clair et précis."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, chain)
        
        with st.spinner("Réflexion..."):
            response = rag_chain.invoke({"input": prompt_text})
            answer = response["answer"]
            
        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.warning("Merci de charger un PDF d'abord !")
