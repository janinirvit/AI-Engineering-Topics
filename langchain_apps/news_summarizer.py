import os
from typing import Optional

from dotenv import load_dotenv
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from newspaper import Article

load_dotenv()

class NewsArticleSummarizer:
    """Class for summarizing news articles using different language models."""

    def __init__(self, api_key: str = None, model_type: str = "openai", model_name: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_type = model_type
        self.model_name = model_name

        if model_type == "openai":
            self.llm = ChatOpenAI(model_name=self.model_name, temperature=0)
        # elif model_type == "ollama":
        #     self.llm = ChatOllama(model=self.model_name, temperature=0, format="json")
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, length_function=len)

    def fetch_article(self, url: str) -> Optional[Article]:
        """Fetches and parses a news article from the given URL."""
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article
        except Exception as e:
            print(f"Failed to fetch article from {url}: {e}")
            return None

    def create_documents(self, text: str) -> list[Document]:
        """Splits the given text into chunks and creates LangChain Document objects."""
        chunks = self.splitter.split_text(text)
        return [Document(page_content=chunk) for chunk in chunks]

    def summarize_article(self, url: str, summary_type: str = "detailed") -> dict:
        """Summarizes the given news article."""
        article = self.fetch_article(url)
        if not article or not article.text:
            return {"error": "No content to summarize."}

        documents = self.create_documents(article.text)
        map_prompt_template = ""
        combine_prompt_template = ""
        if summary_type == "detailed":
            map_prompt_template = """Write a detailed summary of the following text:
            "{text}"
            DETAILED SUMMARY:"""

            combine_prompt_template = """Write a detailed summary of the following text that combines the previous summaries:
            "{text}"
            FINAL DETAILED SUMMARY:"""

        else:
            map_prompt_template = """Write a concise summary of the following text:
            "{text}"
            CONCISE SUMMARY:"""

            combine_prompt_template = """Write a concise summary of the following text that combines the previous summaries:
            "{text}"
            FINAL CONCISE SUMMARY:"""

        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
        combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

        chain = load_summarize_chain(
            self.llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt, verbose=True
        )

        # Generate summary
        summary = chain.invoke(documents)

        return {
            "title": article.title,
            "authors": article.authors,
            "publish_date": article.publish_date,
            "summary": summary,
            "url": url,
            "model_info": {"type": self.model_type, "name": self.model_name},
        }


def main():
    # Example of using both models
    url = "https://www.artificialintelligence-news.com/news/us-china-ai-chip-race-cambricons-first-profit-lands/"

    # Initialize both summarizers
    openai_summarizer = NewsArticleSummarizer(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_type="openai",
        model_name="gpt-4o-mini",
    )

    # ollama_summarizer = NewsArticleSummarizer(
    #     model_type="ollama", model_name="llama3.2"  # or "llama3.2"
    # )

    # Get summaries from both models
    print("\nGenerating OpenAI Summary...")
    openai_summary = openai_summarizer.summarize_article(url, summary_type="detailed")

    # print("\nGenerating Llama Summary...")
    # llama_summary = ollama_summarizer.summarize_article(url, summary_type="detailed")

    # Print results
    for summary, model in [(openai_summary, "OpenAI")]:
        print(f"\n{model} Summary:")
        print("-" * 50)
        print(f"Title: {summary['title']}")
        print(f"Authors: {', '.join(summary['authors'])}")
        print(f"Published: {summary['publish_date']}")
        print(
            f"Model: {summary['model_info']['type']} - {summary['model_info']['name']}"
        )
        print(f"Summary:\n{summary['summary']}")

        # Print first document content
        print("\nFirst Document Content:")
        print(summary["summary"]["input_documents"][0].page_content)

        print("\nSecond Document Content:")
        print(summary["summary"]["input_documents"][1].page_content)


if __name__ == "__main__":
    main()
