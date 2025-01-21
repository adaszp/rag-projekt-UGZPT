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

Based on my experiment **Cosine Similarity** is best for text similarity because it focuses on the angle (direction)
between vectors, making it ideal for comparing text embeddings where magnitude is less important.

In my results, **Cosine values are consistent (e.g., `articles_cosine`: 0.533, `article_pages_cosine`: 0.586)** and
capture relative similarity effectively.

Metrics like **Manhattan** or **Euclidean** are scale-sensitive, while **Dot Product** depends on vector magnitudes,
making them less reliable for normalized text embeddings.

## Choosing embedding model

```json
{
  "all-MiniLM-L6-v2": {
    "articles_all-MiniLM-L6-v2_cosine": {
      "avg_score": 0.5330837845999998,
      "avg_time": 0.018908839999985504
    },
    "articles_all-MiniLM-L6-v2_pages_cosine": {
      "avg_score": 0.5860837879,
      "avg_time": 0.01934328599989385
    }
  },
  "paraphrase-MiniLM-L6-v2": {
    "articles_paraphrase-MiniLM-L6-v2_cosine": {
      "avg_score": 0.5869176734,
      "avg_time": 0.02023351699994237
    },
    "articles_paraphrase-MiniLM-L6-v2_pages_cosine": {
      "avg_score": 0.6343669763,
      "avg_time": 0.019083036000101857
    }
  },
  "distilbert-base-nli-stsb-mean-tokens": {
    "articles_distilbert-base-nli-stsb-mean-tokens_cosine": {
      "avg_score": 0.5150524624000004,
      "avg_time": 0.03314595300007568
    },
    "articles_distilbert-base-nli-stsb-mean-tokens_pages_cosine": {
      "avg_score": 0.5602436013999998,
      "avg_time": 0.03125216900005398
    }
  },
  "all-mpnet-base-v2": {
    "articles_all-mpnet-base-v2_cosine": {
      "avg_score": 0.5747964858,
      "avg_time": 0.055177620999911595
    },
    "articles_all-mpnet-base-v2_pages_cosine": {
      "avg_score": 0.6083208072000003,
      "avg_time": 0.057888600999922345
    }
  },
  "all-distilroberta-v1": {
    "articles_all-distilroberta-v1_cosine": {
      "avg_score": 0.5178937826,
      "avg_time": 0.03305520300000353
    },
    "articles_all-distilroberta-v1_pages_cosine": {
      "avg_score": 0.5619559941999999,
      "avg_time": 0.03313134800000626
    }
  }
}
```
### Analysis
After analyzing the results of different embedding models across two datasets (articles and article pages),
I have concluded that **paraphrase-MiniLM-L6-v2** is the best model for this task.

**paraphrase-MiniLM-L6-v2** consistently performed well with an average score of **0.5869** on the articles dataset and
**0.6344** on the article pages dataset.
Despite a slightly longer average time of **0.0202** and **0.0191** seconds respectively, the improvements in score
outweigh the marginal increase in processing time.

In comparison, **all-MiniLM-L6-v2** showed good performance with an average score of **0.5331** and **0.5861**, but its
score improvement was smaller. **distilbert-base-nli-stsb-mean-tokens** and **all-mpnet-base-v2** showed larger time
differences, with **all-mpnet-base-v2** especially being slower without a significant boost in accuracy.
**all-distilroberta-v1** demonstrated lower scores, confirming it’s less efficient than the others.

Therefore, I chose **paraphrase-MiniLM-L6-v2** as it balances the best performance with relatively low computational
cost, making it the most effective for both datasets.

### Comparison using percentage increase
The formula for percentage increase is:

Percentage Increase = ((New Value - Baseline Value) / Baseline Value) × 100

Here are the percentage comparisons of avg_score and avg_time for each model, with paraphrase-MiniLM-L6-v2 as the
baseline:

| Model                                    | Dataset  | Score Change (%) | Time Change (%) |
|------------------------------------------|----------|------------------|-----------------|
| **all-MiniLM-L6-v2**                     | Articles | -9.17%           | -6.55%          |
|                                          | Pages    | -7.61%           | +1.36%          |
| **distilbert-base-nli-stsb-mean-tokens** | Articles | -12.24%          | +63.82%         |
|                                          | Pages    | -11.68%          | +63.77%         |
| **all-mpnet-base-v2**                    | Articles | -2.07%           | +172.70%        |
|                                          | Pages    | -4.11%           | +203.35%        |
| **all-distilroberta-v1**                 | Articles | -11.76%          | +63.37%         |
|                                          | Pages    | -11.41%          | +73.62%         |

#### Analysis:

**all-MiniLM-L6-v2** showed slight performance drops in both score and time, making it the closest alternative to **paraphrase-MiniLM-L6-v2**.

**distilbert-base-nli-stsb-mean-tokens** showed significant time increases with a larger drop in accuracy, making it less efficient.

**all-mpnet-base-v2** and **all-distilroberta-v1** exhibited large time increases with only modest score improvements, making them inefficient choices.

Thus, paraphrase-MiniLM-L6-v2 remains the best balance between performance and processing time.