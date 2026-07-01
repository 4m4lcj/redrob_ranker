# Redrob Hackathon Ranking Pipeline

## Setup

```bash
pip install -r requirements.txt
```

Copy (or symlink) `candidates.jsonl` into `data/`.

## Explore the data

```bash
cd src
python explore.py --file ../data/candidates.jsonl --limit 500
```

## Run the full pipeline
From the working directory, Run:

```bash
python ./src/ranker.py --candidates ./data/candidates.jsonl --jd ./jd_parsed.json --out ./output/ranked.csv
```

## Project layout

```
redrob_ranker/
  data/               # place candidates.jsonl here
  src/
    keywords.py       # Contains keywords consts for filters.py and embedder.py
    helper.py         # Contains helper functions for filters.py and embedder.py
    filters.py        # Task 1: hard metadata filters
    embedder.py       # Task 2: sentence-transformer embedding + cosine sim
    ranker.py         # Main file

  jd_parsed.json      # parsed JD - filled manually from job_descriptions.docx
  requirements.txt
```
