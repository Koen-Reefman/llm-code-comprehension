from langchain.text_splitter import Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

import os
import time


# Create the embeddings for a given codebase and store them in a Chroma database
def generate_embeddings(project_path: str, chunk_size: int, overlap_size: int):
    try:
        embedding_function = OllamaEmbeddings(
            base_url="http://host.docker.internal:11434",
            model="nomic-embed-text", temperature=0.1)
        embedding_function.embed_query("test")
    except Exception:
        embedding_function = OllamaEmbeddings(
            model="nomic-embed-text", temperature=0.1)

    abs_project_path = f"""{os.path.abspath(
        os.path.join(os.path.dirname(__file__), project_path))}"""

    # Load
    loader = GenericLoader.from_filesystem(
        abs_project_path,
        glob="**/*",
        suffixes=[".java"],
        exclude=[],
        parser=LanguageParser(language=Language.JAVA,
                              parser_threshold=500),
    )
    # https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.parsers.language.language_parser.LanguageParser.html

    documents = loader.load()

    # Split files in chunks
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JAVA, chunk_size=chunk_size, chunk_overlap=overlap_size)
    texts = python_splitter.split_documents(documents)

    Chroma.from_documents(
        texts, embedding_function, persist_directory=f"vector-embedding/nomic_chroma_{chunk_size}_{overlap_size}")

# This method generated multiple collections within one Chroma DB for vector embeddings of different sizes


def generate_multiple(project_path):
    try:
        embedding_function = OllamaEmbeddings(
            base_url="http://host.docker.internal:11434",
            model="nomic-embed-text", temperature=0.1)
        embedding_function.embed_query("test")
    except Exception:
        embedding_function = OllamaEmbeddings(
            model="nomic-embed-text", temperature=0.1)

    abs_project_path = f"""{os.path.abspath(
        os.path.join(os.path.dirname(__file__), project_path))}"""

    # Load
    loader = GenericLoader.from_filesystem(
        abs_project_path,
        glob="**/*",
        suffixes=[".java"],
        exclude=[],
        parser=LanguageParser(language=Language.JAVA,
                              parser_threshold=500),
    )

    # https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.parsers.language.language_parser.LanguageParser.html

    documents = loader.load()

    to_generate = [(400, 100), (1000, 250)]

    for g in to_generate:
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.JAVA, chunk_size=g[0], chunk_overlap=g[1])
        texts = python_splitter.split_documents(documents)

        Chroma.from_documents(texts, embedding_function,
                              collection_name=f"generated_{g[0]}_{g[1]}", persist_directory=f"multiple")


if __name__ == "__main__":
    before = time.time()
    generate_multiple("../path/to/project")
    print(f"Done generating vector embeddings, took {time.time() - before} s")
