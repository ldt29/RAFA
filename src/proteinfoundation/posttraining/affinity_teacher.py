"""Frozen sequence-affinity teacher used by the affinity-first RL reward.

The geometric interface proxy is useful for every structure in the RAFA data,
but it is not a measurement of binding affinity.  This module provides a small
and auditable bridge to experimentally measured antibody affinity datasets.  It
is intentionally a *teacher* rather than a trainable component of RAFA:

* the teacher is fit offline on public affinity measurements;
* the checkpoint is frozen during RL;
* inference only consumes the generated CDR sequence and the fixed antigen
  sequence; and
* all normalization and feature-layout constants are serialized next to the
  weights.

The feature map is deliberately lightweight.  It contains heavy/light CDR
composition, position-specific CDR one-hot features (with a fixed cap), and
antigen composition.  This is not advertised as a state-of-the-art affinity
predictor.  Its role is to add an independent affinity-related signal and to
make calibration failure visible instead of silently calling DockQ affinity.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: index for index, aa in enumerate(AA)}
DEFAULT_MAX_CDR_LEN = 64


DATASET_SPECS: tuple[dict[str, object], ...] = (
    # Binder is log10(Kd) for these sets; lower is stronger, so invert it.
    {"name": "d44", "file": "d44_set.csv", "target": "Binder", "sign": -1.0},
    {"name": "trast", "file": "trast_set.csv", "target": "Binder", "sign": -1.0},
    {
        "name": "anti-Fluorescein",
        "file": "anti-Fluorescein_set.csv",
        "target": "Binder",
        "sign": -1.0,
    },
    {"name": "anti-H1_HA", "file": "anti-H1_HA_set.csv", "target": "Binder", "sign": -1.0},
    # Enrichment ratios are higher for stronger binders.
    {"name": "g6", "file": "g6_set.csv", "target": "enrichment ratio", "sign": 1.0},
    {"name": "anti-HR2_SARS-CoV-2", "file": "anti-HR2_SARS-CoV-2_set.csv", "target": "Binder", "sign": 1.0},
)


def _clean_sequence(value: object) -> str:
    """Keep standard amino acids and make malformed CSV cells harmless."""
    if value is None:
        return ""
    text = str(value).upper()
    return "".join(aa for aa in text if aa in AA_TO_INDEX)


def _cdr_sequences_from_row(row: Mapping[str, object]) -> tuple[str, str]:
    heavy = "".join(_clean_sequence(row.get(key, "")) for key in ("CDRH1", "CDRH2", "CDRH3"))
    light = "".join(_clean_sequence(row.get(key, "")) for key in ("CDRL1", "CDRL2", "CDRL3"))
    # Some scFv rows do not carry split CDR columns.  Falling back to the
    # variable sequence retains a useful signal without pretending to know the
    # exact numbering scheme.
    if not heavy:
        heavy = _clean_sequence(row.get("Heavy_Sequence", ""))
    if not light:
        light = _clean_sequence(row.get("Light_Sequence", ""))
    antigen = _clean_sequence(row.get("Antigen_Sequence", ""))
    return heavy, light, antigen


def feature_dim(max_cdr_len: int = DEFAULT_MAX_CDR_LEN) -> int:
    # composition: heavy CDR, light CDR, antigen; position one-hot: H and L;
    # normalized lengths: H, L, antigen.
    return 3 * len(AA) + 2 * max_cdr_len * len(AA) + 3


def _feature_vector(heavy_cdr: str, light_cdr: str, antigen: str, max_cdr_len: int) -> np.ndarray:
    """Construct the frozen teacher feature vector."""
    out = np.zeros(feature_dim(max_cdr_len), dtype=np.float32)
    cursor = 0

    def composition(sequence: str) -> None:
        nonlocal cursor
        if sequence:
            for aa in sequence:
                index = AA_TO_INDEX.get(aa)
                if index is not None:
                    out[cursor + index] += 1.0 / len(sequence)
        cursor += len(AA)

    composition(heavy_cdr)
    composition(light_cdr)
    composition(antigen)

    def positional(sequence: str) -> None:
        nonlocal cursor
        for position, aa in enumerate(sequence[:max_cdr_len]):
            index = AA_TO_INDEX.get(aa)
            if index is not None:
                out[cursor + position * len(AA) + index] = 1.0
        cursor += max_cdr_len * len(AA)

    positional(heavy_cdr)
    positional(light_cdr)

    for sequence in (heavy_cdr, light_cdr, antigen):
        out[cursor] = min(len(sequence), 512) / 512.0
        cursor += 1
    return out


def _feature_vector_torch(
    heavy_cdr: Tensor,
    heavy_mask: Tensor,
    light_cdr: Tensor,
    light_mask: Tensor,
    antigen: Tensor,
    antigen_mask: Tensor,
    max_cdr_len: int,
) -> Tensor:
    """Torch implementation of :func:`_feature_vector` for rollout scoring."""
    batch_size = heavy_cdr.shape[0]
    device = heavy_cdr.device
    dim = feature_dim(max_cdr_len)
    out = torch.zeros(batch_size, dim, device=device, dtype=torch.float32)
    cursor = 0

    def composition(sequence: Tensor, mask: Tensor) -> None:
        nonlocal cursor
        valid = mask.float()
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        for aa_index in range(len(AA)):
            out[:, cursor + aa_index] = ((sequence == aa_index) * valid).sum(dim=1) / denom[:, 0]
        cursor += len(AA)

    composition(heavy_cdr, heavy_mask)
    composition(light_cdr, light_mask)
    composition(antigen, antigen_mask)

    def positional(sequence: Tensor, mask: Tensor) -> None:
        nonlocal cursor
        n = min(sequence.shape[1], max_cdr_len)
        if n:
            positions = torch.arange(n, device=device).view(1, n).expand(batch_size, -1)
            valid = mask[:, :n]
            for aa_index in range(len(AA)):
                hit = (sequence[:, :n] == aa_index) & valid
                columns = cursor + positions * len(AA) + aa_index
                out.scatter_(1, columns, hit.float())
        cursor += max_cdr_len * len(AA)

    positional(heavy_cdr, heavy_mask)
    positional(light_cdr, light_mask)
    for sequence, mask in ((heavy_cdr, heavy_mask), (light_cdr, light_mask), (antigen, antigen_mask)):
        out[:, cursor] = mask.float().sum(dim=1).clamp(max=512.0) / 512.0
        cursor += 1
    return out


def _row_target(row: Mapping[str, object], spec: Mapping[str, object]) -> float | None:
    try:
        value = float(row[str(spec["target"])])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return float(spec["sign"]) * value


def _stable_group(name: str, heavy: str, light: str, antigen: str) -> int:
    token = f"{name}\n{heavy}\n{light}\n{antigen}".encode("utf-8")
    return int(hashlib.sha1(token).hexdigest()[:8], 16) % 5


def iter_curated_rows(data_dir: str | Path) -> Iterable[dict[str, object]]:
    """Yield normalized rows from the six small curated CSV files."""
    import csv

    root = Path(data_dir)
    for spec in DATASET_SPECS:
        path = root / str(spec["file"])
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                target = _row_target(row, spec)
                if target is None:
                    continue
                heavy, light, antigen = _cdr_sequences_from_row(row)
                if not heavy or not antigen:
                    continue
                yield {
                    "dataset": str(spec["name"]),
                    "heavy_cdr": heavy,
                    "light_cdr": light,
                    "antigen": antigen,
                    "target": target,
                    "group": _stable_group(str(spec["name"]), heavy, light, antigen),
                }


class AffinitySequenceTeacher:
    """Frozen ridge teacher with a pure torch rollout scoring path."""

    def __init__(
        self,
        coef: np.ndarray,
        intercept: float,
        target_center: float,
        target_scale: float,
        max_cdr_len: int = DEFAULT_MAX_CDR_LEN,
        output_center: float = 0.0,
        output_scale: float = 1.0,
    ) -> None:
        self.coef = np.asarray(coef, dtype=np.float32).reshape(-1)
        self.intercept = float(intercept)
        self.target_center = float(target_center)
        self.target_scale = max(float(target_scale), 1e-6)
        self.max_cdr_len = int(max_cdr_len)
        self.output_center = float(output_center)
        self.output_scale = max(float(output_scale), 1e-6)
        if self.coef.size != feature_dim(self.max_cdr_len):
            raise ValueError("teacher coefficient dimension does not match feature layout")

    @classmethod
    def load(cls, path: str | Path) -> "AffinitySequenceTeacher":
        values = np.load(Path(path), allow_pickle=False)
        return cls(
            coef=values["coef"],
            intercept=float(values["intercept"]),
            target_center=float(values["target_center"]),
            target_scale=float(values["target_scale"]),
            max_cdr_len=int(values["max_cdr_len"]),
            output_center=float(values["output_center"]),
            output_scale=float(values["output_scale"]),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            coef=self.coef,
            intercept=np.asarray(self.intercept, dtype=np.float32),
            target_center=np.asarray(self.target_center, dtype=np.float32),
            target_scale=np.asarray(self.target_scale, dtype=np.float32),
            max_cdr_len=np.asarray(self.max_cdr_len, dtype=np.int64),
            output_center=np.asarray(self.output_center, dtype=np.float32),
            output_scale=np.asarray(self.output_scale, dtype=np.float32),
        )

    def predict_numpy(self, rows: Sequence[tuple[str, str, str]]) -> np.ndarray:
        features = np.stack(
            [_feature_vector(h, l, a, self.max_cdr_len) for h, l, a in rows], axis=0
        )
        standardized = (features @ self.coef + self.intercept)
        return standardized * self.target_scale + self.target_center

    def score_torch(self, residue_type: Tensor, batch: Mapping[str, Tensor]) -> Tensor:
        """Return a bounded [0, 1] teacher score for generated sequences.

        ``residue_type`` is the frozen autoencoder decode of a rollout.  The
        antigen sequence is taken from the conditioning batch.  CDR masks are
        used instead of native antibody coordinates, so this signal remains
        available for fully masked generation.
        """
        if residue_type.dim() == 1:
            residue_type = residue_type.unsqueeze(0)
        chain_type = batch["chain_type"]
        # ``cdr_mask`` is the stochastic mixed-masking mask used by the FM
        # conditioner.  The teacher must score the complete decoded CDR, so
        # prefer the immutable native-Cdr annotation when it is available.
        cdr_mask = batch.get("native_cdr_mask", batch["cdr_mask"]).bool()
        if chain_type.dim() == 1:
            chain_type = chain_type.unsqueeze(0)
        if cdr_mask.dim() == 1:
            cdr_mask = cdr_mask.unsqueeze(0)
        valid = batch.get("full_mask", chain_type > 0).bool()
        if valid.dim() == 1:
            valid = valid.unsqueeze(0)

        heavy_mask = cdr_mask & (chain_type == 1) & valid
        light_mask = cdr_mask & (chain_type == 2) & valid
        antigen_mask = (chain_type == 3) & valid

        def gather(mask: Tensor) -> tuple[Tensor, Tensor]:
            width = max(int(mask.sum(dim=1).max().item()), 1)
            positions = torch.arange(mask.shape[1], device=mask.device).view(1, -1)
            rank = (mask.long().cumsum(dim=1) - 1).clamp(min=0)
            selected = torch.zeros(mask.shape[0], width, device=mask.device, dtype=residue_type.dtype)
            selected_mask = torch.zeros(mask.shape[0], width, device=mask.device, dtype=torch.bool)
            for row in range(mask.shape[0]):
                idx = positions[0][mask[row]]
                if idx.numel():
                    count = min(int(idx.numel()), width)
                    selected[row, :count] = residue_type[row, idx[:count]]
                    selected_mask[row, :count] = True
            return selected, selected_mask

        heavy, heavy_valid = gather(heavy_mask)
        light, light_valid = gather(light_mask)
        antigen, antigen_valid = gather(antigen_mask)
        features = _feature_vector_torch(
            heavy, heavy_valid, light, light_valid, antigen, antigen_valid, self.max_cdr_len
        )
        coef = torch.as_tensor(self.coef, device=features.device, dtype=features.dtype)
        standardized = features @ coef + float(self.intercept)
        # A sigmoid maps the standardized regression output to a stable reward
        # range; output_center/scale are fitted from the training predictions.
        return torch.sigmoid((standardized - self.output_center) / self.output_scale)


def fit_teacher(
    data_dir: str | Path,
    *,
    max_cdr_len: int = DEFAULT_MAX_CDR_LEN,
    alpha: float = 10.0,
    seed: int = 5,
) -> tuple[AffinitySequenceTeacher, dict[str, object]]:
    """Fit the teacher and return the model plus a calibration summary."""
    from scipy import sparse
    from scipy.stats import kendalltau, pearsonr, spearmanr
    from sklearn.linear_model import Ridge

    rows = list(iter_curated_rows(data_dir))
    if len(rows) < 100:
        raise ValueError(f"only {len(rows)} usable affinity rows; refusing to fit teacher")
    features = np.stack(
        [_feature_vector(row["heavy_cdr"], row["light_cdr"], row["antigen"], max_cdr_len) for row in rows],
        axis=0,
    ).astype(np.float32)
    targets = np.asarray([float(row["target"]) for row in rows], dtype=np.float32)
    groups = np.asarray([int(row["group"]) for row in rows], dtype=np.int64)
    train_mask = groups != 0
    valid_mask = ~train_mask
    center = float(targets[train_mask].mean())
    scale = float(targets[train_mask].std())
    scale = max(scale, 1e-6)
    target_z = (targets - center) / scale
    # Ridge's lsqr solver consumes sparse input and avoids a several-GB dense
    # design matrix when the HR2 mutational scan is included.
    model = Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr")
    model.fit(sparse.csr_matrix(features[train_mask]), target_z[train_mask])
    pred_z = model.predict(sparse.csr_matrix(features))
    pred = pred_z * scale + center
    valid_pred = pred[valid_mask]
    valid_target = targets[valid_mask]
    if len(valid_target) > 2:
        spearman = float(spearmanr(valid_target, valid_pred).statistic)
        pearson = float(pearsonr(valid_target, valid_pred).statistic)
        kendall = float(kendalltau(valid_target, valid_pred).statistic)
        # Bootstrap the frozen grouped holdout, not the training rows.  This
        # gives the paper an uncertainty interval without treating correlated
        # training examples as independent evidence.
        rng = np.random.default_rng(int(seed) + 7919)
        bootstrap_spearman = []
        bootstrap_pearson = []
        for _ in range(500):
            sample = rng.integers(0, len(valid_target), size=len(valid_target))
            target_sample = valid_target[sample]
            pred_sample = valid_pred[sample]
            bootstrap_spearman.append(float(spearmanr(target_sample, pred_sample).statistic))
            bootstrap_pearson.append(float(pearsonr(target_sample, pred_sample).statistic))
        spearman_ci = [
            float(np.nanquantile(bootstrap_spearman, 0.025)),
            float(np.nanquantile(bootstrap_spearman, 0.975)),
        ]
        pearson_ci = [
            float(np.nanquantile(bootstrap_pearson, 0.025)),
            float(np.nanquantile(bootstrap_pearson, 0.975)),
        ]
        # A concordance probability is an interpretable rank readout and is
        # less sensitive to the score's sigmoid calibration than Pearson.
        order = np.argsort(valid_target, kind="mergesort")
        target_sorted = valid_target[order]
        pred_sorted = valid_pred[order]
        concordant = 0
        discordant = 0
        for left in range(len(target_sorted) - 1):
            target_delta = target_sorted[left + 1 :] - target_sorted[left]
            pred_delta = pred_sorted[left + 1 :] - pred_sorted[left]
            comparable = target_delta != 0
            concordant += int(((target_delta[comparable] * pred_delta[comparable]) > 0).sum())
            discordant += int(((target_delta[comparable] * pred_delta[comparable]) < 0).sum())
        pair_total = concordant + discordant
        concordance = float(concordant / pair_total) if pair_total else float("nan")
    else:
        spearman = float("nan")
        pearson = float("nan")
        kendall = float("nan")
        spearman_ci = [float("nan"), float("nan")]
        pearson_ci = [float("nan"), float("nan")]
        concordance = float("nan")
    output_center = float(np.median(pred_z[train_mask]))
    output_scale = float(np.std(pred_z[train_mask]))
    output_scale = max(output_scale, 0.25)
    teacher = AffinitySequenceTeacher(
        coef=np.asarray(model.coef_, dtype=np.float32),
        intercept=float(model.intercept_),
        target_center=center,
        target_scale=scale,
        max_cdr_len=max_cdr_len,
        output_center=output_center,
        output_scale=output_scale,
    )
    by_dataset: dict[str, dict[str, float | int]] = {}
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        mask = np.asarray([str(row["dataset"]) == dataset for row in rows]) & valid_mask
        if int(mask.sum()) < 3:
            continue
        by_dataset[dataset] = {
            "n": int(mask.sum()),
            "spearman": float(spearmanr(targets[mask], pred[mask]).statistic),
            "pearson": float(pearsonr(targets[mask], pred[mask]).statistic),
        }
    summary = {
        "teacher": "sequence_ridge_v1",
        "feature_dim": feature_dim(max_cdr_len),
        "max_cdr_len": max_cdr_len,
        "alpha": alpha,
        "seed": seed,
        "n_rows": len(rows),
        "n_train": int(train_mask.sum()),
        "n_valid": int(valid_mask.sum()),
        "target_direction": "higher_is_stronger",
        "valid_spearman": spearman,
        "valid_pearson": pearson,
        "valid_kendall": kendall,
        "valid_spearman_bootstrap95": spearman_ci,
        "valid_pearson_bootstrap95": pearson_ci,
        "valid_pairwise_concordance": concordance,
        "bootstrap_replicates": 500,
        "by_dataset_valid": by_dataset,
        "files": [str(spec["file"]) for spec in DATASET_SPECS],
        "note": "Grouped holdout uses a stable antibody-antigen sequence hash; this is calibration evidence, not a claim of general affinity prediction.",
    }
    return teacher, summary


__all__ = [
    "AffinitySequenceTeacher",
    "DATASET_SPECS",
    "DEFAULT_MAX_CDR_LEN",
    "feature_dim",
    "fit_teacher",
    "iter_curated_rows",
]
