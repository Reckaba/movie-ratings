"""Rebuild data/movies.db from the MovieLens 20M files.

The MovieLens license does not allow redistributing the data, so it is not in this
repository. Download MovieLens 20M from https://grouplens.org/datasets/movielens/20m/
(or Kaggle: grouplens/movielens-20m-dataset), put the ratings and movies files into
data/ (ratings.csv or rating.csv, movies.csv or movie.csv), then run:

    python build_db.py

The `ratings` table is a random sample of 500,000 ratings (fixed seed) joined with
titles, genres and release year; the `movies` table is the full catalog.
"""
import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
SEED = 42


def find(*names: str) -> Path:
    for name in names:
        if (DATA / name).exists():
            return DATA / name
    raise FileNotFoundError(f"Put one of {names} into {DATA} (see the docstring of build_db.py)")


def sample_ratings(path: Path, n: int, seed: int = SEED) -> pd.DataFrame:
    """Uniform random sample of n rows, read in chunks so 20M rows never sit in memory."""
    with open(path, encoding="utf-8") as f:
        total = sum(1 for _ in f) - 1
    rng = np.random.default_rng(seed)
    keep = min(1.0, 1.05 * n / total)  # oversample slightly, then trim to exactly n
    chunks = [chunk[rng.random(len(chunk)) < keep] for chunk in pd.read_csv(path, chunksize=2_000_000)]
    return pd.concat(chunks).sample(n=n, random_state=seed).sort_index()


def main(n: int) -> None:
    movies = pd.read_csv(find("movies.csv", "movie.csv"))
    ratings = sample_ratings(find("ratings.csv", "rating.csv"), n)

    if pd.api.types.is_numeric_dtype(ratings["timestamp"]):  # GroupLens files use Unix time
        ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    movies["year"] = pd.to_numeric(movies["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
    ratings = ratings.merge(movies, on="movieId", how="left")

    with sqlite3.connect(DATA / "movies.db") as con:
        ratings[["userId", "movieId", "rating", "timestamp", "title", "genres", "year"]].to_sql(
            "ratings", con, if_exists="replace", index=False)
        movies[["movieId", "title", "genres"]].to_sql("movies", con, if_exists="replace", index=False)
    print(f"data/movies.db: {len(ratings):,} sampled ratings, {len(movies):,} movies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=500_000, help="number of ratings to sample")
    main(parser.parse_args().n)
