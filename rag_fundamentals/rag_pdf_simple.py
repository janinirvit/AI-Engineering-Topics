import os
import uuid

import chromadb
import streamlit as st
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
api_key = os.getenv("OPENAI_API_KEY")
nomic_api_base = os.getenv("NOMIC_API_BASE")


class SimpleModelSelector:
    def __init__(self):
        self.llm_model = {"openai": "GPT-4", "ollama": "Llama3"}
        self.embedding_models = {
            "openai": {"name": "OpenAI Embedding", "dimensions": 1536, "model_name": "text-embedding-3-small"},
            "chroma": {"name": "Chroma Embedding", "dimensions": 384, "model_name": None},
            "nomic": {"name": "Nomic Embedding", "dimensions": 768, "model_name": "nomic-embed-text"},
        }

    def select_models(self):

        st.sidebar.title("Select Models")

        llm = st.sidebar.radio("Select LLM Model", list(self.llm_model.keys()), format_func=lambda x: self.llm_model[x])

        embedding = st.sidebar.radio(
            "Select Embedding Model",
            list(self.embedding_models.keys()),
            format_func=lambda x: self.embedding_models[x]["name"],
        )
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

            if end < len(text):
                last_period = chunk.rfind(".")
                if last_period != -1:
                    chunk = chunk[: last_period + 1]
                    end = start + last_period + 1

            chunks.append({"id": f"{uuid.uuid4()}", "text": chunk, "metadata": {"source": pdf_file.name}})
            start = end
        return chunks


class SimpleRAGSystem:
    def __init__(self, llm_model="openai", embedding_model="openai"):
        self.llm_model = llm_model
        self.embedding_model = embedding_model

        self.db = chromadb.PersistentClient(path="./chromadb")

        self.setup_embedding_functions()  # Assuming this is the intended method to set up embeddings

        if llm_model == "openai":
            self.llm = OpenAI(api_key=api_key)
        else:
            self.llm = OpenAI(base_url=nomic_api_base, api_key="ollama")

        self.collection = self.setup_collection()

    def setup_embedding_functions(self):
            try:
                if self.embedding_model == "openai":
                    self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=api_key, model_name="text-embedding-3-small"
                    )
                elif self.embedding_model == "chroma":
                    self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
                elif self.embedding_model == "nomic":
                    self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=api_key, api_base=os.getenv("NOMIC_API_BASE"), model_name="nomic-embed-text"
                    )
                else:
                    self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            except Exception as e:
                print(f"Error setting up embedding function: {e}")
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

    def setup_collection(self):
        print(f"Setting up collection for embedding model: {self.embedding_model}")

        collection_name = f"documents_{self.embedding_model}"

        try:
            collection = self.db.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
            st.info(f"Existing collection: {collection_name}")
            return collection
        except Exception as e:
            print(f"Error setting up collection: {e}")
            collection = self.db.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"model": self.embedding_model},
            )
            st.info(f"Created new collection: {collection_name}")
        return collection

    def get_embedding_info(self):
        """Get information about the embedding function."""
        model_selector = SimpleModelSelector()
        model_info = model_selector.embedding_models[self.embedding_model]
        return {
            "name": model_info.get("name"),
            "dimensions": model_info.get("dimensions"),
            "model": self.embedding_model if hasattr(self, "embedding_model") else None,
        }

    def add_documents(self, chunks):
        """Add documents to ChromaDB collection.

        Args:
            chunks (list): A list of dictionaries containing chunk information.
        """
        try:
            if not self.collection:
                self.collection = self.setup_collection()
            self.collection.add(
                documents=[chunk["text"] for chunk in chunks],
                metadatas=[chunk["metadata"] for chunk in chunks],
                ids=[chunk["id"] for chunk in chunks],
            )
            return True
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False

    def query_documents(self, query, top_k=2):
        try:
            if not self.collection:
                raise ValueError("Collection is not set up.")

            results = self.collection.query(query_texts=[query], n_results=top_k)
            return results
        except Exception as e:
            print(f"Error querying documents: {e}")
            return []

    def generate_response(self, query, context):
        try:
            prompt = f"""
            Based on the following context, please answer the question.
            If you can't find the answer in the context, say so or I don't know.
            Context:
            {context}
            Question:
            {query}
            Answer:
            """

            response = (
                self.llm.chat.completions.create(
                    model="gpt-4o-mini" if self.llm_model == "openai" else "llama3.2",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                )
                .choices[0]
                .message.content
            )
            return response
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return f"Error generating response: {e}"




def streamlit_app():

    st.title("RAG Pipeline for Space Facts")

    # initialize session state
    if "processed_files" not in st.session_state:
        st.session_state["processed_files"] = set()
    if "current_embedding_model" not in st.session_state:
        st.session_state["current_embedding_model"] = None
    if "rag_system" not in st.session_state:
        st.session_state["rag_system"] = None

    model_selector = SimpleModelSelector()
    llm_model, embedding_model = model_selector.select_models()

    if embedding_model != st.session_state.current_embedding_model:
        st.session_state.processed_files.clear()
        st.session_state.rag_system = None
        st.session_state.current_embedding_model = embedding_model
        st.warning("Embedding model changed. Cleared processed files and RAG system.")

    try:
        # Initialize RAG system if not already done
        if st.session_state.rag_system is None:
            st.session_state.rag_system = SimpleRAGSystem(llm_model, embedding_model)

        embedding_info = st.session_state.rag_system.get_embedding_info()
        st.sidebar.info(f"Embedding Info: {embedding_info}")
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return

    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
    if pdf_file:
        if pdf_file.name not in st.session_state.processed_files:
            processor = SimplePDFProcessor()
            with st.spinner("Processing PDF..."):
                try:
                    text = processor.read_pdf(pdf_file)
                    chunks = processor.create_chunks(text, pdf_file)

                    if st.session_state.rag_system.add_documents(chunks):
                        st.session_state.processed_files.add(pdf_file.name)
                        st.success(f"Successfully proccessed {pdf_file.name} to RAG system.")
                except Exception as e:
                    st.error(f"Error processing PDF with processor: {e}")

        if st.session_state.processed_files:
            st.markdown("### Processed Files")
            st.subheader("Query your documents")

        query = st.text_input("Ask a question")

        if query:
            with st.spinner("Generating response..."):
                results = st.session_state.rag_system.query_documents(query)
                if results and results["documents"]:
                    # Generate Response
                    response = st.session_state.rag_system.generate_response(query, results["documents"])
                    if response:
                        #Display resonse
                        st.markdown("### Answer")
                        st.write(response)

                        with st.expander("View Source Passages", expanded=False):
                            for idx, doc in enumerate(results["documents"][0], 1):
                                st.markdown(f"**Source Passage {idx}:**")
                                st.info(doc)

    else:
        st.info("Please upload a PDF file to start.")


def main():
    streamlit_app()


if __name__ == "__main__":
    main()
