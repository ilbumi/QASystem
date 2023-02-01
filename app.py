import logging
import os

import uvicorn
from fastapi import FastAPI
from haystack.document_stores.milvus import MilvusDocumentStore
from haystack.nodes import EmbeddingRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline

from download_data import download_data
from settings import *
from schema import Answer, QueryRequest, QueryResponse

logging.basicConfig(
    format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING
)
logging.getLogger("haystack").setLevel(logging.INFO)
if not os.path.exists(DOC_DIR):
    download_data()


def get_pipeline():
    """Construct whole pipeline for QA."""
    document_store = get_document_store()
    return ExtractiveQAPipeline(get_reader(), get_retriever(document_store))


def get_document_store() -> MilvusDocumentStore:
    """Construct document store.
    This is the memory of the app.
    MilvusDocumentStore can perform fast similiarity search using vector embeddings of the documents.
    This fast search aims to narrow down the search scope.
    """

    document_store = MilvusDocumentStore(
        DOC_DB,
        host="milvus",
        embedding_dim=RETRIEVER_EMB_SIZE,
        index_type="ANNOY",  # ANNOY is fast
    )
    return document_store


def get_reader() -> FARMReader:
    """Construct QA model itself."""
    return FARMReader(model_name_or_path=QA_MODEL, use_gpu=USE_GPU)


def get_retriever(document_store: MilvusDocumentStore) -> EmbeddingRetriever:
    """Construct Retriever.
    Retriever performs the quick search (see `get_document_store`)
    """
    retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model=RETRIEVER_MODEL, use_gpu=USE_GPU
    )
    document_store.update_embeddings(retriever)
    return retriever


# application settings
app = FastAPI(
    title="Haystack REST API",
    debug=True,
)
query_pipeline = get_pipeline()


def simplify_dict(prediction) -> QueryResponse:
    return QueryResponse(
        question=prediction["query"],
        results=[
            Answer(answer=answer.answer, context=answer.context, score=answer.score)
            for answer in prediction["answers"]
        ],
    )


@app.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
def query(request: QueryRequest):
    if request.params is None:
        request.params = {"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
    prediction = query_pipeline.run(
        query=request.question, params=request.params, debug=True
    )
    return simplify_dict(prediction)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
