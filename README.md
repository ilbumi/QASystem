# Haystack QA example

This code deploys a docker container with a question-answering system based on [Haystack](https://haystack.deepset.ai/overview/quick-start).

## Running

```bash
docker compose up -d 
```

This deploys all the necessary containers:

- [Milvus](https://milvus.io/) to store documents and their embeddings.
- FastAPI endpoint

### Running parameters

The main parameters are stored in `settings.py`:

- `DOC_DIR`: directory where the raw txt files will be downloaded
- `RETRIEVER_MODEL`: [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=downloads) model name to retrieve relevant documents before performing QA
- `RETRIEVER_EMB_SIZE`: embedding size of the RETRIEVER_MODEL
- `QA_MODEL`: [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=question-answering&sort=downloads) model name to retrieve answers
- `USE_GPU`: whether to use GPU or not

Also, you may want to select another [PyTorch base image](https://hub.docker.com/r/pytorch/pytorch/tags) (by changing `BASE_IMAGE` in `docker-compose.yml`) but beware of dependencies breaks.

## Tweaks

Other changes demand code modification.
You can change:

- Document store (FAISS, ElasticSearch, and other), default is Milvus store
- Documents chunk sizes (see preprocessor in `download_data.py`)
- Retrieval method (BM25, TF-IDF), default is embeddings similarity retrieval

Modify `download_data.py` to populate the database with your documents in txt format.
