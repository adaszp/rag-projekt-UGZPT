# RAG UGZPT

## Setting up Qdrant vector database

To set up database you need to have Docker installed on your local machine.

To install and use Qdrant vector database:
1. Download latest Qdrant Docker image:
```bash
docker pull qdrant/qdrant
```
2. Run docker compose script from project root to set up ready to go container
```bash
docker compose up --wait
```

## Sample setup

1. Run [process_data.py](./process_data.py) to process all json data into Qdrant collections.


## Choosing distance metric

Choosing a distance metric was done, by using generated queries to match points in Qdrant vector database,
and by getting average score.

```json
{
  "articles_cosine": 0.5860837879,
  "article_pages_cosine": 0.61496635025,
  "articles_euclid": 0.9605274951999996,
  "article_pages_euclid": 0.9037213943000001,
  "articles_dot": 0.5330837845999998,
  "article_pages_dot": 0.5860837879,
  "articles_manhattan": 14.96285151,
  "article_pages_manhattan": 14.062382705000006
}
```

Based on my experiment **Cosine Similarity** is best for text similarity because it focuses on the angle (direction) between vectors, making it ideal for comparing text embeddings where magnitude is less important. 

In my results, **Cosine values are consistent (e.g., `articles_cosine`: 0.533, `article_pages_cosine`: 0.586)** and capture relative similarity effectively.

Metrics like **Manhattan** or **Euclidean** are scale-sensitive, while **Dot Product** depends on vector magnitudes, making them less reliable for normalized text embeddings.