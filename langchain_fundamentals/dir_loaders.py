from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("./data", glob="**/*.txt")
documents = loader.load()
for i, doc in enumerate(documents):
    print(f"Document {i + 1}:")
    print(doc)
    print("\n")
