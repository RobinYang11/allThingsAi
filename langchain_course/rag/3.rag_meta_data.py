import os
from langchain.text_splitter import CharacterTextSplitter

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="deepseek-r1:7b",
    # model="llama3",
    base_url="http://localhost:11434",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

if not os.path.exists(persistent_directory):
    print("Chunking books directory ...")
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path"
        )

    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path, encoding="utf-8")
        book_docs = loader.load()
        for doc in book_docs:
            doc.metadata = {"source": book_file}
            documents.append(doc)

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separator="。",
        length_function=len,
    )
    docs = text_splitter.split_documents(documents)
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
else:
    print("Vecto store already exists. No need to initialize.")
