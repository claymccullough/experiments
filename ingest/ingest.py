# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import shutil

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv('.env')

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma

"""
GLOBALS
"""
INGEST_BASE_PATH = "./embeddings"
INGEST_FILE_PATH = os.environ.get("INGEST_FILE_PATH")
MODEL_NAME = os.environ.get("EMBED_MODEL_NAME")
BASE_URL = os.environ.get("EMBED_BASE_URL")
INGEST_NAME = os.environ.get("INGEST_NAME")
# CHUNK_SIZES = [512, 1024, 2048, 4096]
# CHUNK_SIZES = [1024, 2048, 4096]
# CHUNK_SIZES = [1024, 2048, 4096]
# CHUNK_OVERLAPS = [0, 50, 100]
# CHUNK_OVERLAPS = [100]
# CHUNK_VARIATIONS = [(chunk_size, chunk_overlap) for chunk_size in CHUNK_SIZES for chunk_overlap in CHUNK_OVERLAPS]
# CHUNK_VARIATION_FILES = [f'{INGEST_BASE_PATH}/{INGEST_NAME}x{chunk_size}x{chunk_overlap}' for chunk_size, chunk_overlap in
#                          CHUNK_VARIATIONS]

print(MODEL_NAME, BASE_URL)


def metadata_func(record: dict, metadata: dict) -> dict:
    """
    "title": "Metadata",
    "source": "https://data-platform-docs.limeade.work/",
    "page_content": "name: Metadata\nschema_version: \ndbt_version: 1.7.4\ngenerated_at: 2024-02-13T21:11:38.056393Z\ninvocation_id: 6cac193c-305c-463d-9177-e0f9eb2f83c4\nmodel_count: 249\nsource_count: 137"
  },
  {
    "title": "model.limeade_lakehouse.sil_ai_identity_events",
    "source": "https://data-platform-docs.limeade.work/#!/model/model.limeade_lakehouse.sil_ai_identity_events",
    "name": "model.limeade_lakehouse.sil_ai_identity_events",
    "schema": "silver_app_insights",
    "owner": "root",
    "upstream_models": [
      "source.limeade_lakehouse.bronze_ai_limeadeone_identity.event",
      "source.limeade_lakehouse.bronze_ai_limeadeone_identity_ds.events"
    ],
    "downstream_models": [
      "model.limeade_lakehouse.gold_stg_user_events_telemetry_identity",
      "test.limeade_lakehouse.not_null_sil_ai_identity_events_identity_event_id.939505fb74"
    ],
    "bytes": 111039368,
    "rows": 727576,
    :param record:
    :param metadata:
    :return:
    """
    return {
        'source': record.get('source'),
        'title': record.get('title'),
        # 'name': record.get('name', 'N/A'),
        # 'schema': record.get('schema', 'N/A'),
        # 'owner': record.get('owner', 'N/A'),
        # 'upstream_models': record.get('upstream_models', 'N/A'),
        # 'downstream_models': record.get('downstream_models', 'N/A'),
        # 'bytes': record.get('bytes', 'N/A'),
        # 'rows': record.get('rows', 'N/A'),
    }


if __name__ == '__main__':
    print('STARTING INGEST')

    print('JSONLOADER...')
    # json_loader = JSONLoader(
    #     file_path=INGEST_FILE_PATH,
    #     jq_schema=".[]",
    #     content_key="page_content",
    #     metadata_func=metadata_func
    # )
    folder_path = f'{INGEST_BASE_PATH}/{INGEST_NAME}'
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)

    json_loader = JSONLoader(
        file_path=INGEST_FILE_PATH,
        jq_schema=".[]",
        content_key="page_content",
        metadata_func=metadata_func
    )
    embedding_function = OllamaEmbeddings(model=MODEL_NAME, base_url=BASE_URL)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    # docs = text_splitter.split_documents(json_loader.load())
    docs = json_loader.load()
    print(f'LOADING {len(docs)} DOCUMENTS...')
    db = Chroma.from_documents(docs, embedding_function, persist_directory=f'{INGEST_BASE_PATH}/{INGEST_NAME}')

    print('DONE')


    # print(f'CREATING EMBEDDINGS, CHUNK_SIZES: {CHUNK_SIZES}, CHUNK_OVERLAPS: {CHUNK_OVERLAPS}...')

    # for chunk_size, chunk_overlap in (pbar := tqdm(CHUNK_VARIATIONS)):
    #     folder_name = f'{INGEST_BASE_PATH}/{INGEST_NAME}x{chunk_size}x{chunk_overlap}'
    #     if os.path.isdir(folder_name):
    #         pbar.update(1)
    #         continue
    #     os.mkdir(folder_name)
    #
    #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    #     docs = text_splitter.split_documents(json_loader.load())
    #
    #     # Load into chroma
    #     pbar.set_description(f"{chunk_size}x{chunk_overlap}->{len(docs)}")
    #     embedding_function = OllamaEmbeddings(model=MODEL_NAME, base_url=BASE_URL)
    #     db = Chroma.from_documents(docs, embedding_function, persist_directory=folder_name)
    #     pbar.update(1)
