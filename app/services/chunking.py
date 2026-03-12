from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=512,
    chunk_overlap=100
)

def split_documents(documents: List[Document]) -> List[Document]:
    if not documents:
        raise ValueError("No documents to split.")
    chunks = text_splitter.split_documents(documents)
    return chunks