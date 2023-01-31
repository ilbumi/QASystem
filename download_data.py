import os
from glob import glob
from pathlib import Path

import wget
from haystack.document_stores.milvus import MilvusDocumentStore
from haystack.nodes import PreProcessor, TextConverter

from settings import *


def download_data():
    os.makedirs(DOC_DIR, exist_ok=True)
    wget.download(
        "https://www.gutenberg.org/cache/epub/11/pg11.txt", out=DOC_DIR
    )  # Alice in Wonderland
    wget.download(
        "https://www.gutenberg.org/cache/epub/6130/pg6130.txt", out=DOC_DIR
    )  # The Iliad

    # prepare data store
    converter = TextConverter()
    processor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",  # splitting is essential!
        split_length=384,
        split_respect_sentence_boundary=True,
        split_overlap=64,
    )
    document_store = MilvusDocumentStore(
        DOC_DB, host="milvus", embedding_dim=RETRIEVER_EMB_SIZE
    )
    for txt_file in glob(f"{DOC_DIR}/*.txt"):
        docs = converter.convert(file_path=Path(txt_file), meta=None)
        for doc in docs:
            more_docs = processor.process(doc)
            print(len(more_docs))
            document_store.write_documents(more_docs)
