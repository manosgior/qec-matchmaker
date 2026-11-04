"""Training, saving, loading and predicting with Matchmaker's models.

Gradient-boosted trees (scikit-learn HistGradientBoosting) on the circuit features of FEATURE_NAMES, never
on the code's name, so that unseen codes are predicted from their circuits. Features the training table
does not fill are left out, and missing values elsewhere stay NaN, which the trees handle. The
per-decoder models also take the decoder, as a categorical input.

- regime: per circuit, P(best LER > 0.5)
- ler: per circuit, quantiles of log10 best LER, on circuits whose best decode is measured
- regret: per decoder, median of log10(LER / best LER), on circuits with at least two measured decoders
- time: per decoder, quantiles of log10 ms per shot. A timed-out decode is right-censored: its time per
  shot exceeds timeout_s over a nominal first batch of 100 shots. After a first fit on the decodes with a
  time, each censored decode joins a second fit at max(its predicted median, that bound).
"""
import datetime
import json
import math
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .decoders import DECODERS, applicable
from .features import FEATURE_NAMES, FEATURE_VERSION

RATED = ("measured", "low statistics", "upper bound")
ABOVE = 0.5
LER_FLOOR = 1e-8
FIRST_BATCH = 100
PARAMS = dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
              early_stopping=False)


class Constant:
    """Stands in for a model whose training labels leave nothing to learn (no rows, or one class)"""

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row of features per circuit of the training table, indexed by circuit key"""
    return df.groupby("circuit_key", sort=False)[[c for c in FEATURE_NAMES if c in df.columns]].first()


def circuit_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Per circuit with a rate: its best LER (an upper bound counts as its bound), the status of the decode
    that gave it, and whether it is above threshold; indexed by circuit key"""
    rated = df[df.status.isin(RATED)]
    rated = rated.assign(rate=np.where(rated.errors > 0, rated.ler, rated.ler_high)).dropna(subset=["rate"])
    best = rated.loc[rated.groupby("circuit_key").rate.idxmin()]
    return pd.DataFrame({"best_ler": best.rate.to_numpy(float), "best_status": best.status.to_numpy(),
                         "above": (best.rate > ABOVE).to_numpy()}, index=pd.Index(best.circuit_key, name="circuit_key"))


def regret_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Measured decodes of circuits with at least two measured decoders, with log10(LER / best measured LER)"""
    m = df[(df.status == "measured") & (df.ler > 0)]
    m = m[m.groupby("circuit_key").decoder.transform("size") >= 2]
    return m.assign(regret=np.log10(m.ler / m.groupby("circuit_key").ler.transform("min")))


def timeout_bound_ms(timeout_s):
    """Lower bound on the ms per shot of a decode that timed out before its first batch finished"""
    return np.asarray(timeout_s, float) * 1000 / FIRST_BATCH


def _matrix(frame, features):
    return frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce").to_numpy(float)


def _with_decoder(X, decoders, vocab):
    """X with the decoder as a last, categorical column; decoders outside vocab are NaN"""
    index = {d: i for i, d in enumerate(vocab)}
    return np.column_stack([X, np.array([index.get(d, np.nan) for d in decoders], dtype=float)])


def _qname(prefix, q):
    return f"{prefix}_q{round(100 * q):02d}"


def _regressor(quantile, categorical, random_state):
    return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, random_state=random_state,
                                         categorical_features=categorical, **PARAMS)


def _fit_quantiles(X, y, quantiles, categorical, random_state):
    if len(y) == 0:
        return {q: Constant(math.nan) for q in quantiles}
    return {q: _regressor(q, categorical, random_state).fit(X, y) for q in quantiles}


def _predict_quantiles(models, X, quantiles):
    """Quantile predictions (rows x quantiles), sorted along each row so that they never cross"""
    return np.sort(np.column_stack([models[q].predict(X) for q in quantiles]), axis=1)


def _p_above(model, X):
    return model.predict(X) if isinstance(model, Constant) else model.predict_proba(X)[:, 1]


def train(df: pd.DataFrame, quantiles=(0.1, 0.5, 0.9), random_state=0, manifest: dict | None = None) -> "ModelBundle":
    """Train all models on a training table (dataset.py); manifest, if given, is kept in the bundle's meta"""
    quantiles = sorted(float(q) for q in quantiles)
    if 0.5 not in quantiles:
        raise ValueError("quantiles must include 0.5")
    features = [c for c in FEATURE_NAMES if c in df.columns and df[c].notna().any()]
    circuits = circuit_features(df)
    labels = circuit_labels(df)

    X = _matrix(circuits.loc[labels.index], features)
    above = labels.above.to_numpy(int)
    if len(np.unique(above)) < 2:
        regime = Constant(float(above[0]) if len(above) else math.nan)
    else:
        regime = HistGradientBoostingClassifier(random_state=random_state, **PARAMS).fit(X, above)

    measured = labels.best_status.to_numpy() == "measured"
    y = np.log10(np.clip(labels.best_ler.to_numpy()[measured], LER_FLOOR, 1))
    ler = _fit_quantiles(X[measured], y, quantiles, None, random_state)

    reg = regret_rows(df)
    regret_vocab = sorted(reg.decoder.unique())
    categorical = [False] * len(features) + [True]
    regret = _fit_quantiles(_with_decoder(_matrix(reg, features), reg.decoder, regret_vocab), reg.regret.to_numpy(),
                            [0.5], categorical, random_state)[0.5]

    timed = df[df.ms_per_shot > 0]
    censored = df[(df.status == "timeout") & (df.timeout_s > 0) & ~(df.ms_per_shot > 0)]
    time_vocab = sorted(set(timed.decoder) | set(censored.decoder))
    Xt = _with_decoder(_matrix(timed, features), timed.decoder, time_vocab)
    yt = np.log10(timed.ms_per_shot.to_numpy(float))
    time = _fit_quantiles(Xt, yt, quantiles, categorical, random_state)
    if len(censored) and len(timed):
        Xc = _with_decoder(_matrix(censored, features), censored.decoder, time_vocab)
        yc = np.maximum(time[0.5].predict(Xc), np.log10(timeout_bound_ms(censored.timeout_s)))
        time = _fit_quantiles(np.vstack([Xt, Xc]), np.concatenate([yt, yc]), quantiles, categorical, random_state)

    meta = {
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "feature_version": FEATURE_VERSION,
        "features": features,
        "quantiles": quantiles,
        "decoders": {"regret": regret_vocab, "time": time_vocab},
        "params": PARAMS,
        "random_state": random_state,
        "training": {"rows": len(df), "circuits": len(circuits), "regime_circuits": len(labels),
                     "above_share": float(above.mean()) if len(above) else math.nan, "ler_circuits": int(measured.sum()),
                     "regret_decodes": len(reg), "time_decodes": len(timed), "time_censored": len(censored)},
        "manifest": manifest,
    }
    return ModelBundle({"regime": regime, "ler": ler, "regret": regret, "time": time}, meta)


class ModelBundle:
    """The trained models, and in meta what they were trained on"""

    def __init__(self, models: dict, meta: dict):
        self.models, self.meta = models, meta

    @property
    def features(self) -> list:
        return self.meta["features"]

    @property
    def quantiles(self) -> list:
        return self.meta["quantiles"]

    def decoders(self) -> list:
        """The decoders seen in training, in the order of decoders.DECODERS"""
        seen = set(self.meta["decoders"]["regret"]) | set(self.meta["decoders"]["time"])
        return [d for d in DECODERS if d in seen] + sorted(seen - set(DECODERS))

    def predict(self, features, decoders=None) -> pd.DataFrame:
        """Predictions per circuit and decoder. features: a frame with one row per circuit (FEATURE_NAMES
        columns, optionally `circuit_key`, else the index is the key), or one circuit's features dict.
        decoders: the decoders to predict for (default: those seen in training); a model that did not see
        a decoder in training predicts NaN for it"""
        frame = pd.DataFrame([features]) if isinstance(features, dict) else pd.DataFrame(features)
        keys = frame["circuit_key"].to_numpy() if "circuit_key" in frame.columns else frame.index.to_numpy()
        decoders = list(self.decoders() if decoders is None else decoders)
        n, k = len(frame), len(decoders)
        X = _matrix(frame, self.features)
        q, models = self.quantiles, self.models

        ler = 10 ** _predict_quantiles(models["ler"], X, q)
        Xd, dec = np.repeat(X, k, axis=0), np.tile(np.asarray(decoders, dtype=object), n)
        regret_vocab, time_vocab = self.meta["decoders"]["regret"], self.meta["decoders"]["time"]
        regret = 10 ** np.maximum(models["regret"].predict(_with_decoder(Xd, dec, regret_vocab)), 0)
        regret[~np.isin(dec, regret_vocab)] = np.nan
        ms = 10 ** _predict_quantiles(models["time"], _with_decoder(Xd, dec, time_vocab), q)
        ms[~np.isin(dec, time_vocab)] = np.nan

        out = pd.DataFrame({"circuit_key": np.repeat(keys, k), "decoder": dec,
                            "p_above": np.repeat(_p_above(models["regime"], X), k)})
        for j, quantile in enumerate(q):
            out[_qname("ler", quantile)] = np.repeat(ler[:, j], k)
        out["regret_q50"] = regret
        out["ler_decoder_q50"] = out.ler_q50 * regret
        for j, quantile in enumerate(q):
            out[_qname("ms", quantile)] = ms[:, j]
        everything = frame.reindex(columns=FEATURE_NAMES).apply(pd.to_numeric, errors="coerce").astype(float)
        fits = [applicable(f) for f in everything.to_dict("records")]
        out["applicable"] = [fit.get(d, True) for fit in fits for d in decoders]
        return out

    def save(self, out_dir: str) -> None:
        """Write models.joblib (the estimators) and meta.json"""
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.models, os.path.join(out_dir, "models.joblib"))
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    @classmethod
    def load(cls, model_dir: str) -> "ModelBundle":
        """Read a saved bundle; warns when it was trained with another scikit-learn or feature version"""
        with open(os.path.join(model_dir, "meta.json")) as f:
            meta = json.load(f)
        if meta.get("sklearn_version") != sklearn.__version__:
            warnings.warn(f"models trained with scikit-learn {meta.get('sklearn_version')}, running "
                          f"{sklearn.__version__}: predictions may differ or fail", stacklevel=2)
        if meta.get("feature_version") != FEATURE_VERSION:
            warnings.warn(f"models trained on feature version {meta.get('feature_version')}, features are now "
                          f"version {FEATURE_VERSION}: retrain", stacklevel=2)
        return cls(joblib.load(os.path.join(model_dir, "models.joblib")), meta)
