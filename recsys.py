"""Top-N movie recommenders and their offline evaluation.

Models share one interface: `fit(X)` on a binary user x item matrix, then
`score(X_users)` returns a dense score for every item for those users.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def temporal_split(ratings: pd.DataFrame, test_share: float = 0.2, min_ratings: int = 10):
    """Hold out each user's most recent `test_share` of ratings as the test set.

    Users with fewer than `min_ratings` ratings stay entirely in the training set.
    """
    r = ratings.sort_values(["userId", "timestamp", "movieId"], kind="stable")
    n = r.groupby("userId")["movieId"].transform("size")
    from_end = r.groupby("userId").cumcount(ascending=False)
    n_test = np.maximum(1, np.round(n * test_share))
    is_test = (n >= min_ratings) & (from_end < n_test)
    return r[~is_test], r[is_test]


class Interactions:
    """Binary user x item matrix built from training ratings, with id <-> index maps."""

    def __init__(self, train: pd.DataFrame):
        self.users = pd.Index(np.sort(train["userId"].unique()))
        self.items = pd.Index(np.sort(train["movieId"].unique()))
        rows = self.users.get_indexer(train["userId"])
        cols = self.items.get_indexer(train["movieId"])
        X = sp.csr_matrix((np.ones(len(train), dtype=np.float32), (rows, cols)),
                          shape=(len(self.users), len(self.items)))
        X.sum_duplicates()
        X.data[:] = 1.0
        self.X = X


class Popularity:
    """Recommend the most-rated movies the user has not seen."""

    def fit(self, X):
        self.counts_ = np.asarray(X.sum(axis=0), dtype=float).ravel()
        return self

    def score(self, X_users):
        return np.tile(self.counts_, (X_users.shape[0], 1))


class _GramModel:
    """Base for item-item models trained on movies with at least `min_count` ratings."""

    def __init__(self, min_count: int = 5):
        self.min_count = min_count

    def _gram(self, X):
        counts = np.asarray(X.sum(axis=0)).ravel()
        self.keep_ = np.flatnonzero(counts >= self.min_count)
        self.n_items_ = X.shape[1]
        Xk = X[:, self.keep_]
        return (Xk.T @ Xk).toarray().astype(np.float64), counts[self.keep_]

    def _item_scores(self, X_users, weights):
        scores = np.full((X_users.shape[0], self.n_items_), -np.inf)
        scores[:, self.keep_] = X_users[:, self.keep_] @ weights
        return scores


class ItemKNN(_GramModel):
    """Item-based collaborative filtering: cosine similarity with shrinkage."""

    def __init__(self, shrink: float = 20.0, min_count: int = 5):
        super().__init__(min_count)
        self.shrink = shrink

    def fit(self, X):
        G, counts = self._gram(X)
        self.sim_ = G / (np.sqrt(np.outer(counts, counts)) + self.shrink)
        np.fill_diagonal(self.sim_, 0.0)
        return self

    def score(self, X_users):
        return self._item_scores(X_users, self.sim_)


class EASE(_GramModel):
    """Embarrassingly Shallow Autoencoder (Steck, 2019): closed-form item-item weights."""

    def __init__(self, lam: float = 200.0, min_count: int = 5):
        super().__init__(min_count)
        self.lam = lam

    def fit(self, X):
        G, _ = self._gram(X)
        G[np.diag_indices_from(G)] += self.lam
        P = np.linalg.inv(G)
        self.B_ = -P / np.diag(P)
        np.fill_diagonal(self.B_, 0.0)
        return self

    def score(self, X_users):
        return self._item_scores(X_users, self.B_)


def evaluate(model, data: Interactions, test: pd.DataFrame, k: int = 10, batch_size: int = 1000) -> dict:
    """Precision, recall and NDCG at k over users with test ratings, plus catalog coverage.

    A recommendation counts as a hit if the user rated that movie in the test period.
    Movies the user rated in training are never recommended. Test ratings of movies
    that do not appear in training cannot be recommended by any model and are dropped.
    """
    test = test[test["movieId"].isin(data.items) & test["userId"].isin(data.users)]
    rows = data.users.get_indexer(test["userId"])
    cols = data.items.get_indexer(test["movieId"])
    truth = sp.csr_matrix((np.ones(len(test), dtype=bool), (rows, cols)), shape=data.X.shape)
    eval_rows = np.unique(rows)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    precision, recall, ndcg, recommended = [], [], [], set()
    for start in range(0, len(eval_rows), batch_size):
        batch = eval_rows[start:start + batch_size]
        seen = data.X[batch]
        scores = model.score(seen)
        scores[seen.nonzero()] = -np.inf
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
        top = np.take_along_axis(top, order, axis=1)

        relevant = truth[batch].toarray()
        hits = np.take_along_axis(relevant, top, axis=1)
        n_relevant = relevant.sum(axis=1)
        ideal = np.cumsum(discounts)[np.minimum(n_relevant, k) - 1]

        precision.extend(hits.sum(axis=1) / k)
        recall.extend(hits.sum(axis=1) / n_relevant)
        ndcg.extend((hits * discounts).sum(axis=1) / ideal)
        recommended.update(np.unique(top).tolist())

    return {
        f"Precision@{k}": float(np.mean(precision)),
        f"Recall@{k}": float(np.mean(recall)),
        f"NDCG@{k}": float(np.mean(ndcg)),
        "Coverage": len(recommended) / data.X.shape[1],
        "Users": len(eval_rows),
    }
