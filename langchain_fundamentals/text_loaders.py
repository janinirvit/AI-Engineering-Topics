import pprint
import re

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def clean_text(text):
    # Remove unwanted characters (e.g., digits, special characters)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Convert to lowercase
    text = text.lower()

    return text


documents = TextLoader("./doc/dream.txt").load()

# cleaned_documents = [clean_text(doc.page_content) for doc in documents]

# print(cleaned_documents)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

texts = [clean_text(chunk.page_content) for chunk in chunks]

# Load embeddings for the cleaned text chunks
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create the retirever from loaded embeddings and documents
retriever = FAISS.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 2})

# query = "What did Martin Luther King Jr. dream about?"
query = "Give me summary of speech in bullet points"
docs = retriever.invoke(query)

pprint.pprint(f"Retrieved documents: {docs}:")

prompt = ChatPromptTemplate.from_template(
    "Please use the following documents \n{docs} to answer the following question:: {query}"
)

model = ChatOpenAI(model="gpt-4o-mini")

chain = prompt | model | StrOutputParser()

response = chain.invoke({"docs": docs, "query": query})
print(f"Response::: {response}")
