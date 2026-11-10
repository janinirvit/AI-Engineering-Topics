from langchain_community.document_loaders import PyPDFLoader

pdf = PyPDFLoader("./doc/linux-manual.pdf")
documents = pdf.load()
for i, doc in enumerate(documents):
    print(f"Document {i + 1}:")
    print(doc)
