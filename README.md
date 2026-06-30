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

1. Fill in `jd_parsed.json` with the parsed job description fields.
2. Run:

```bash
cd src
python ranker.py --candidates ../data/candidates.jsonl --jd ../jd_parsed.json --output ../output/ranked.csv
```

## Project layout

```
redrob_ranker/
  data/               # place candidates.jsonl here
  src/
    filters.py        # Task 1: hard metadata filters
    embedder.py       # Task 2: sentence-transformer embedding + cosine sim
    ranker.py         # orchestrator
    explore.py        # EDA script (run first)
  jd_parsed.json      # parsed JD — fill manually
  requirements.txt
```
