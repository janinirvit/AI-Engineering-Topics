from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_loader = TextLoader("./doc/dream.txt")
documents = text_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20, length_function=len)
split_documents = text_splitter.split_documents(documents)
for i, doc in enumerate(split_documents):
    print(f"Split {i + 1}:")
    print(doc)
    print("\n")
