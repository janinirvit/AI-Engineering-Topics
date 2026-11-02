import csv
import os
from turtle import st
from turtle import st
import uuid

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
api_key = os.getenv("OPENAI_API_KEY")

class SimpleModelSelector:
    def __init__(self):
        self.llm_model = { "openai": "GPT-4", "ollama" : "Llama3"}
        self.embedding_model = {
                            "openai": {
                                "name" : "OpenAI Embedding",
                                "dimensions" : 1536,
                                "model_name" : "text-embedding-3-small"
                            },
                            "chroma": {
                                "name" : "Chroma Embedding",
                                "dimensions" : 384,
                                "model_name" : None
                            },
                            "nomic": {
                                "name" : "Nomic Embedding",
                                "dimensions" : 768,
                                "model_name" : "nomic-embed-text"
                            }
                        }
           
    def select_models(self):
        st.sidebar.title("Select Models")
        
        llm = st.sidebar.radio("Select LLM Model", list(self.llm_model.keys()), format_func=lambda x: self.llm_model[x])
        embedding = st.sidebar.radio("Select Embedding Model", list(self.embedding_model.keys()), format_func=lambda x: self.embedding_model[x]["name"])
        return llm, embedding

class SimplePDFProcessor:
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def read_pdf(self, file_path):
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def create_chunks(self, text, pdf_file):
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            
            if start > 0:
                start -= self.chunk_overlap
            
            chunk = text[start:end]
            
            if end > len(text):
                last_period = chunk.rfind(".")
                if last_period != -1:
                    chunk = chunk[:last_period+1]
                    end = start + last_period + 1
                    
            chunks.append({
                "id": f"{uuid.uuid4()}",
                "text": chunk,
                "metadata": {
                    "source": pdf_file.name
                }
            })
            start = end
        return chunks
    
    
class SimpleRAGSystem:
    def __init__(self, llm_model, embedding_model):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        
        self.db = chromadb.PersistentClient(path="./chromadb")
        
        self.setup_embedding_functions()  # Assuming this is the intended method to set up embeddings

        if llm_model is "openai":
            self.llm = OpenAI(api_key=api_key)
        else:
            pass
        
        self.collection = self.setup_collection()
        
    # def add_documents(self, chunks):
    #     for chunk in chunks:
    #         self.collection.add(
    #             documents=[chunk["text"]],
    #             metadatas=[chunk["metadata"]],
    #             ids=[chunk["id"]]
    #         )
            
    def setup_collection(self):
        return self.db.get_or_create_collection(
            name=f"documents_{self.embedding_model}",
            embedding_function=self.embedding_function,
            metadata={"model": self.embedding_model}
        )
        
    def setup_embedding_functions(self):
        if self.embedding_model == "openai":
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-3-small"
            )
        elif self.embedding_model == "chroma":
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        elif self.embedding_model == "nomic":
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                api_base=os.getenv("NOMIC_API_BASE"),
                model_name="nomic-embed-text"
            )
# Load the data
class EmbeddingModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if self.model_type == "openai":
            self.client = OpenAI(api_key=api_key)
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-3-small"
            )
        elif self.model_type == "chroma":
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        elif self.model_type == "nomic":    
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                api_base=os.getenv("NOMIC_API_BASE"),
                model_name="nomic-embed-text"
            )


class LLMModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if self.model_type == "openai":
            self.client = OpenAI(api_key=api_key)
            self.model_name = "gpt-4o-mini"
        else:
            # Initialize other model here
            pass
        
    def generate_completion(self, message):
        # Accept either a plain string prompt or a fully-formed list of
        # chat messages (e.g. [{"role": "system", ...}, {"role": "user", ...}]).
        messages = message if isinstance(message, list) else [{"role": "user", "content": message}]
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating completion: {e}")
            return f"Error generating response: {e}"
        
        
def select_model(model_type="openai"):
    print(f"Selecting model of type: {model_type}")
    print("1. OpenAI GPT-4")
    print("2. Ollama Llama")
    
    while True:
        choice = input("Select a model (1-2): ").strip()
        if choice == "1":
            model_type = "openai"
            break
        elif choice == "2":
            model_type = "ollama"
            break
        else:
            print("Invalid choice. Please try again.")
    print("\nSelect Embedding Model")
    print("1. OpenAI")
    print("2. Chroma")
    print("3. Nomic")

    while True:
        embedding_choice = input("Select an embedding model (1-3): ").strip()
        if embedding_choice == "1":
            embedding_model = "openai"
            break
        elif embedding_choice == "2":
            embedding_model = "chroma"
            break
        elif embedding_choice == "3":
            embedding_model = "nomic"
            break
        else:
            print("Invalid choice. Please try again.")

    return model_type, embedding_model

def generate_csv(output_path="space_fact.csv"):
    facts = [
        {"id": 1, "fact": "The Sun accounts for about 99.86% of the mass in the entire solar system."},
        {"id": 2, "fact": "A day on Venus is longer than its year."},
        {"id": 3, "fact": "Neutron stars can spin at a rate of 600 rotations per second."},
        {"id": 4, "fact": "There are more stars in the universe than grains of sand on all of Earth's beaches."},
        {"id": 5, "fact": "Jupiter's Great Red Spot is a giant storm that has been raging for over 300 years."},
        {"id": 6, "fact": "One million Earths could fit inside the Sun."},
        {"id": 7, "fact": "Light from the Sun takes about 8 minutes and 20 seconds to reach Earth."},
        {"id": 8, "fact": "Saturn's rings are made mostly of ice particles with a smaller amount of rocky debris and dust."},
        {"id": 9, "fact": "The footprints left by astronauts on the Moon will likely remain for millions of years."},
        {"id": 10, "fact": "Black holes can warp space and time due to their immense gravitational pull."},
    ]

    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["id", "fact"])
        writer.writeheader()
        writer.writerows(facts)

    print(f"Generated CSV file with {len(facts)} facts at: {output_path}")
    return output_path

def load_csv(input_path="space_fact.csv"):
    df = pd.read_csv(input_path)
    documents = df["fact"].tolist()
    print(f"Loaded {len(documents)} documents from CSV.")
    for doc in documents:
        print(f" - {doc}")
    return documents

def setup_chromadb(documents, embedding_model):

    client = chromadb.Client()
    try:
        client.delete_collection(name="space_facts")
    except:  # noqa: S110
        pass
    
    collection = client.create_collection(name="space_facts", embedding_function=embedding_model.embedding_function)
    
    collection.add(
        documents=documents,
        ids=[str(i) for i in range(len(documents))]
    )

    print(f"ChromaDB collection created with {len(documents)} documents.")
    return collection

def find_related_chunks(query, collection, top_k=2):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    print("related chunks found:")
    for chunk in results["documents"][0]:
        print(f" - {chunk}")
    
    return list(zip(results["documents"][0], 
                    (results["metadatas"][0] if results["metadatas"][0] else [{}] * len(results["documents"][0]))))

def augment_prompt(query, related_chunks):
    context = "\n".join([chunk[0] for chunk in related_chunks])
    augmented_prompt = f"{query}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    print("Augmented prompt:")
    print(augmented_prompt)
    
    return augmented_prompt

def rag_pipeline(query, collection, llm_model, top_k=2):
    print(f"Running RAG pipeline for query: {query}")
    related_chunks = find_related_chunks(query, collection, top_k=top_k)
    augmented_prompt = augment_prompt(query, related_chunks)
    
    response = llm_model.generate_completion(
        [
            {"role": "system", "content": "You are a helpful assistant who can questions about space but only answer based on the provided sources/documents given."},
            {"role": "user", "content": augmented_prompt}
        ]
    )
    
    print("Generated response:")
    print(response)
    
    reference = "\n".join([chunk[0] for chunk in related_chunks])
    print("Reference context:")
    print(reference)
    return response, related_chunks, augmented_prompt

def streamlit_app():
    import streamlit as st
    st.set_page_config(page_title="RAG Pipeline for Space Facts", layout="wide")
    st.title("RAG Pipeline for Space Facts")
    
    st.sidebar.title("Model Selection")
    llm_type = st.sidebar.radio("Select LLM model", ["openai", "ollama"],
                                format_func=lambda x: "OpenAI GPT-4" if x == "openai" else "Ollama")
    
    
    embedding_type = st.sidebar.radio(
        "Select Embedding model", ["openai", "chroma", "nomic"],
        format_func=lambda x: "OpenAI Embedding" if x == "openai" else ("Chroma Default" if x == "chroma" else "Nomic")
    )
    
    if "initialized" not in st.session_state:
        st.session_state["initialized"] = False
        generate_csv()
        st.session_state.facts = load_csv()
        
        st.session_state.llm_model = LLMModel(llm_type)
        st.session_state.embedding_model = EmbeddingModel(embedding_type)
        
        #setup ChromaDB collection
        documents = st.session_state.facts
        st.session_state.collection = setup_chromadb(documents, st.session_state.embedding_model)
        st.session_state.initialized = True
    
    if(st.session_state.llm_model.model_type != llm_type or
       st.session_state.embedding_model.model_type != embedding_type):
        st.session_state.llm_model = LLMModel(llm_type)
        st.session_state.embedding_model = EmbeddingModel(embedding_type)
        documents = st.session_state.facts
        st.session_state.collection = setup_chromadb(documents, st.session_state.embedding_model)
    
    
    
    
    query = st.text_input("Enter your query about space:",
                          placeholder="Ask a question about space")
    
    if query:
        with st.spinner("Generating response..."):
            response, references , augment_prompt = rag_pipeline(query, st.session_state.collection, st.session_state.llm_model)

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Response")
                st.write(response)
            with col2:
                st.subheader("Reference used")
                for ref in references:
                    st.write(f" - {ref[0]}")
            with st.expander("TECHNICAL DETAILS", expanded=False):
                st.markdown("### Augmented Prompt")
                st.code(augment_prompt)
                
                st.markdown("### Model Configuration")
                st.code(f"LLM Model: {st.session_state.llm_model.model_type}\nEmbedding Model: {st.session_state.embedding_model.model_type}")
        
def main():
    streamlit_app()
        
if __name__ == "__main__":
    main()
    