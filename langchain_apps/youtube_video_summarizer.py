# pip install yt_dlp - extract youtube videos
# pip install -q git+https://github.com/openai/whisper.git - Speech recognition

# Required to install also:
# MacOS (requires https://brew.sh/):
# brew install ffmpeg

# ubuntu
# sudo apt install ffmpeg

# windows
# https://www.hostinger.com/tutorials/how-to-install-ffmpeg - install ffmpeg on Windows
# FFmpeg is an open-source, command-line tool for working with audio and video files.

import os
from typing import Dict

import whisper
import yt_dlp
from dotenv import load_dotenv
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


class EmbeddingModel:
    def __init__(self, model_type: str = "openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.embedding_fn = OpenAIEmbeddings(
                model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        elif model_type == "chroma":
            self.embedding_fn = HuggingFaceEmbeddings()
        elif model_type == "nomic":
            from langchain.embeddings import OllamaEmbeddings

            self.embedding_fn = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
        else:
            raise ValueError(f"Unsupported embedding type: {model_type}")


class LLMModel:
    def __init__(self, model_type: str = "openai", model_name="gpt-4o-mini"):
        self.model_type = model_type
        self.model_name = model_name
        if model_type == "openai":
            self.llm = ChatOpenAI(model=model_name, openai_api_key=os.getenv("OPENAI_API_KEY"))
        elif model_type == "ollama":
            self.llm = ChatOllama(
                model=model_name, base_url="http://localhost:11434", temperature=0, format="json", timeout=120
            )
        else:
            raise ValueError(f"Unsupported LLM type: {model_type}")


class YoutubeVideoSummarizer:
    def __init__(self, llm_type="openai", llm_model_name="gpt-4", embedding_type="openai"):
        self.llm_model = LLMModel(model_type=llm_type, model_name=llm_model_name)
        self.embedding_model = EmbeddingModel(model_type=embedding_type)
        self.whisper_model = whisper.load_model("base")

    def get_model_info(self) -> Dict:
        """Return current model configuration"""
        return {
            "llm_type": self.llm_model.model_type,
            "llm_model": self.llm_model.model_name,
            "embedding_type": self.embedding_model.model_type,
        }

    def download_video(self, url: str, output_path: str = "downloads/%(title)s.%(ext)s") -> tuple[str, str]:
        print(f"Downloading video from {url} and extract autdio to path {output_path}")

        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": output_path,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_path = ydl.prepare_filename(info).replace(".webm", ".mp3")
            video_title = info.get("title", "unknown")
            return audio_path, video_title

    def transcribe_audio(self, audio_path: str) -> str:
        print(f"Transcribing audio from {audio_path}")
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]

    def create_documents(self, text: str, video_title: str) -> list[Document]:
        print(f"Creating documents for video title: {video_title}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
        )

        docs = text_splitter.split_text(text)
        return [Document(page_content=chunk, metadata={"title": video_title}) for chunk in docs]

    def create_vector_store(self, documents: list[Document]) -> Chroma:
        print(f"Creating vector store for {len(documents)} documents")
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model.embedding_fn,
            collection_name=f"youtube_summary_{self.embedding_model.model_type}",
        )

    def generate_summary(self, documents: list[Document]) -> str:
        print(f"Generating summary for {len(documents)} documents")

        map_prompt = ChatPromptTemplate.from_template(
            """Write a concise summary of the following transcript section:
            "{text}"
            CONCISE SUMMARY:"""
        )

        combine_prompt = ChatPromptTemplate.from_template(
            """Write a detailed summary of the following video transcript sections:
            "{text}"

            Include:
            - Main topics and key points
            - Important details and examples
            - Any conclusions or call to action

            DETAILED SUMMARY:"""
        )

        summary_chain = load_summarize_chain(
            llm=self.llm_model.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True,
        )

        response = summary_chain.invoke(documents)
        return response

    def setup_qa_chain(self, vector_store: Chroma):

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm_model.llm, retriever=vector_store.as_retriever(), memory=memory, verbose=True
        )

    def process_video(self, url: str) -> dict:
        try:
            print(f"Processing video: {url}")
            os.makedirs("downloads", exist_ok=True)

            # Download video and Process
            audio_path, video_title = self.download_video(url, output_path="downloads/%(title)s.%(ext)s")
            transcript = self.transcribe_audio(audio_path)
            documents = self.create_documents(transcript, video_title)
            summary = self.generate_summary(documents)
            vector_store = self.create_vector_store(documents)
            qa_chain = self.setup_qa_chain(vector_store)

            # clean up
            os.remove(audio_path)

            return {
                "summary": summary,
                "qa_chain": qa_chain,
                "title": video_title,
                "full_transcript": transcript,
            }
        except Exception as e:
            print(f"Error processing video {url}: {e}")
            return None


def main():
    # use these urls for testing
    # urls = [
    #     "https://www.youtube.com/watch?v=v48gJFQvE1Y&ab_channel=BrockMesarich%7CAIforNonTechies",
    #     "https://www.youtube.com/watch?v=XwZkNaTYBQI&ab_channel=TheGadgetGameShow%3AWhatTheHeckIsThat%3F%21",
    # ]

    # Get Model Preperences
    print("\nAvailable LLM Models:")
    print("1. OpenAI GPT-4")
    print("2. Ollama Llama3.2")
    llm_choice = input("Choose LLM model (1/2): ").strip()

    print("\nAvailable Embeddings:")
    print("1. OpenAI")
    print("2. Chroma Default")
    print("3. Nomic (via Ollama)")
    embedding_choice = input("Choose embeddings (1/2/3): ").strip()

    # Configure model settings
    llm_type = "openai" if llm_choice == "1" else "ollama"
    llm_model_name = "gpt-4o-mini" if llm_choice == "1" else "llama3.2"

    if embedding_choice == "1":
        embedding_type = "openai"
    elif embedding_choice == "2":
        embedding_type = "chroma"
    else:
        embedding_type = "nomic"

    try:
        # Initialize the YouTubeVideoSummarizer with the chosen LLM model
        summarizer = YoutubeVideoSummarizer(
            llm_type=llm_type,
            llm_model_name=llm_model_name,
            embedding_type=embedding_type,
        )

        # display configuration
        model_info = summarizer.get_model_info()
        print("\nCurrent Configuration:")
        print(f"LLM: {model_info['llm_type']} ({model_info['llm_model']})")
        print(f"Embeddings: {model_info['embedding_type']}")

        # Process video
        url = input("\nEnter YouTube URL: ")
        print("\nProcessing video...")
        result = summarizer.process_video(url)

        if result:
            print(f"\nVideo Title: {result['title']}")
            print("\nSummary:")
            print(result["summary"])

            print("\nYou can now ask questions about the video (type 'quit' to exit)")
            while True:
                query = input("\nYour question: ").strip()
                if query.lower() == "quit":
                    break
                if query:
                    response = result["qa_chain"].invoke({"question": query})
                    print("\nAnswer:", response["answer"])

            # Option to see full transcript
            if input("\nWant to see the full transcript? (y/n): ").lower() == "y":
                print("\nFull Transcript:")
                print(result["full_transcript"])

    except Exception as e:
        print(f"Error initializing summarizer: {e}")


if __name__ == "__main__":
    main()
