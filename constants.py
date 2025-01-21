CONTAINER_URL = 'localhost'
CONTAINER_PORT = 6333

BASE_COLLECTION_NAME = "articles"

MODEL_NAME_ALL_MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_NAME_PARAPHRASE_MINILM = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
MODEL_NAME_DISTILBERT = 'sentence-transformers/distilbert-base-nli-stsb-mean-tokens'
MODEL_NAME_MPNET = 'sentence-transformers/all-mpnet-base-v2'
MODEL_NAME_DISTILROBERTA = 'sentence-transformers/all-distilroberta-v1'

METRIC_DOT = "Dot"
METRIC_COSINE = "Cosine"
METRIC_EUCLID = "Euclid"
METRIC_MANHATTAN = "Manhattan"