#!/usr/bin/env python3
"""
DrugBank category cohesion / retrieval study

- Robust SMILES -> Mol parsing (sanitize exceptions handled)
- Standardized molecule normalization (salt removal / fragment chooser / optional neutralization)
- Deduplication by canonical SMILES
- Multiple fingerprints: Morgan (bit + count), FeatureMorgan, RDKit, MACCS, AtomPair, TopologicalTorsion
- Multiple similarity metrics (RDKit DataStructs) where applicable
- Descriptor panel expanded + multiple NN metrics (cosine/euclidean/manhattan/correlation/mahalanobis)
- Efficient top-K retrieval (no full NxN matrices); RDKit bulk similarities when available
- Graded relevance using category-set Jaccard (plus binary relevance)
- IR metrics: Precision@K, Recall@K, MAP@K, nDCG@K
- Micro + Macro averaging (macro over categories)
- Random baseline
- Bootstrap confidence intervals
"""

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances


# =========================
# Configuration
# =========================

CSV_PATH = "drugbank-short-database.csv"
SEP = "|"

SMILES_COL = "smiles"
CATEGORIES_COL = "categories"
CATEGORY_SEP = "#"

TOP_K_LIST = [1, 5, 10, 20, 50]

# Fingerprints
MORGAN_RADIUS_LIST = [1, 2, 3]
MORGAN_BITS_LIST = [1024, 2048, 4096]

# Bootstrap
BOOTSTRAP_B = 200
BOOTSTRAP_SEED = 7

# Output
OUT_TABLE = "retrieval_metrics_by_representation.csv"

# Normalization options
REMOVE_SALTS = False
NEUTRALIZE = False  # neutralization can sometimes distort; enable if you want
KEEP_LARGEST_FRAGMENT = False
DEDUP_CANONICAL_SMILES = False

RANDOM_BASELINE = True


# =========================
# Utility: molecule parsing & normalization
# =========================

def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return None
        return mol
    except Exception:
        return None


def normalize_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Salt stripping / fragment choose / optional neutralization."""
    if mol is None:
        return None
    try:
        m = Chem.Mol(mol)

        if REMOVE_SALTS:
            remover = rdMolStandardize.SaltRemover()
            m = remover.StripMol(m, dontRemoveEverything=True)

        if KEEP_LARGEST_FRAGMENT:
            chooser = rdMolStandardize.LargestFragmentChooser()
            m = chooser.choose(m)

        if NEUTRALIZE:
            # Uncharger removes formal charges when possible
            uncharger = rdMolStandardize.Uncharger()
            m = uncharger.uncharge(m)

        # Re-sanitize to ensure consistency
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


# =========================
# Category processing & relevance
# =========================

def parse_categories(x) -> List[str]:
    if pd.isna(x):
        return []
    s = str(x)
    if not s.strip():
        return []
    return [c for c in s.split(CATEGORY_SEP) if c]


def jaccard_set(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def dcg_at_k(rels: Sequence[float], k: int) -> float:
    """DCG with rels already in retrieved order (not necessarily sorted)."""
    s = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        # log2(i+1)
        s += (2.0**rel - 1.0) / math.log2(i + 1.0)
    return s


def ndcg_at_k(rels: Sequence[float], ideal_rels: Sequence[float], k: int) -> float:
    d = dcg_at_k(rels, k)
    id_ = dcg_at_k(ideal_rels, k)
    return (d / id_) if id_ > 0 else 0.0


def average_precision_at_k(binary_rels: Sequence[int], total_relevant: int, k: int) -> float:
    """
    AP@K for binary relevance.
    Divide by min(total_relevant, K) (standard for truncated AP).
    """
    if total_relevant <= 0:
        return 0.0
    denom = min(total_relevant, k)
    hit = 0
    s = 0.0
    for i, r in enumerate(binary_rels[:k], start=1):
        if r:
            hit += 1
            s += hit / i
    return s / denom


# =========================
# Descriptor panel
# =========================

DESCRIPTOR_FNS = [
    Descriptors.MolWt,
    Descriptors.MolLogP,
    Descriptors.TPSA,
    Descriptors.NumHDonors,
    Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds,
    Descriptors.RingCount,
    Descriptors.NumAromaticRings,
    rdMolDescriptors.CalcFractionCSP3,
    Descriptors.HeavyAtomCount,
    Descriptors.NHOHCount,
    Descriptors.NOCount,
]


def build_descriptor_matrix(mols: List[Chem.Mol]) -> np.ndarray:
    X = np.array([[fn(m) for fn in DESCRIPTOR_FNS] for m in mols], dtype=float)
    X = StandardScaler().fit_transform(X)
    return X


# =========================
# Fingerprints
# =========================

def fp_morgan_bit(m: Chem.Mol, radius: int, nbits: int, use_features: bool = False):
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits, useFeatures=use_features)

def fp_morgan_count(m: Chem.Mol, radius: int, use_features: bool = False):
    # Returns a SparseIntVect (count-based)
    return AllChem.GetMorganFingerprint(m, radius, useFeatures=use_features)

def fp_rdk_bit(m: Chem.Mol):
    return Chem.RDKFingerprint(m)

def fp_maccs(m: Chem.Mol):
    return MACCSkeys.GenMACCSKeys(m)

def fp_atompair(m: Chem.Mol):
    # count-based sparse vector
    return rdMolDescriptors.GetHashedAtomPairFingerprint(m)

def fp_toptorsion(m: Chem.Mol):
    # count-based sparse vector
    return rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(m)


# =========================
# Similarity functions
# =========================

@dataclass
class SimilaritySpec:
    name: str
    sim_fn: Callable
    bulk_fn: Optional[Callable] = None  # if available: (fp_i, fps) -> list[float]


def get_similarity_specs_for_fp_kind(fp_kind: str) -> List[SimilaritySpec]:
    """
    fp_kind: "bit" or "count"
    Many RDKit similarity fns work on both ExplicitBitVect and SparseIntVect.
    Bulk fns exist for some metrics and bit vectors; for counts you may still use them in practice
    (RDKit supports bulk Tanimoto/Dice/Cosine for SparseIntVect too in most builds).
    """
    specs = [
        SimilaritySpec("Tanimoto", DataStructs.TanimotoSimilarity, DataStructs.BulkTanimotoSimilarity),
        SimilaritySpec("Dice", DataStructs.DiceSimilarity, DataStructs.BulkDiceSimilarity),
        SimilaritySpec("Cosine", DataStructs.CosineSimilarity, DataStructs.BulkCosineSimilarity),
        # Extra metrics (no bulk in typical RDKit):
        SimilaritySpec("Kulczynski", DataStructs.KulczynskiSimilarity, None),
        SimilaritySpec("Asymmetric", DataStructs.AsymmetricSimilarity, None),
        SimilaritySpec("BraunBlanquet", DataStructs.BraunBlanquetSimilarity, None),
        SimilaritySpec("Sokal", DataStructs.SokalSimilarity, None),
        SimilaritySpec("McConnaughey", DataStructs.McConnaugheySimilarity, None),
        SimilaritySpec("RogotGoldberg", DataStructs.RogotGoldbergSimilarity, None),
        SimilaritySpec("Russel", DataStructs.RusselSimilarity, None),
        SimilaritySpec("OnBit", DataStructs.OnBitSimilarity, None),
    ]
    # For count vectors, OnBit is less meaningful (still defined, but interpret with caution).
    if fp_kind == "count":
        # Keep it for completeness, but you may comment it out if it behaves oddly for your RDKit build.
        return specs
    return specs


def topk_neighbors_by_similarity(
    fps: List,
    sim_spec: SimilaritySpec,
    k: int,
    exclude_self: bool = True
) -> List[np.ndarray]:
    """
    For each i, compute top-k neighbor indices based on similarity to all j.
    No NxN matrix stored.
    """
    n = len(fps)
    out = []

    for i in range(n):
        # Try bulk if provided; fall back if RDKit signature doesn't match (common for count FPs + Cosine)
        sims = None
        if sim_spec.bulk_fn is not None:
            try:
                sims = np.array(sim_spec.bulk_fn(fps[i], fps), dtype=float)
            except Exception:
                sims = None

        if sims is None:
            sims = np.array([sim_spec.sim_fn(fps[i], fps[j]) for j in range(n)], dtype=float)

        if exclude_self:
            sims[i] = -np.inf

        if k < n:
            idx = np.argpartition(sims, -k)[-k:]
        else:
            idx = np.arange(n)

        idx = idx[np.argsort(sims[idx])[::-1]]
        out.append(idx[:k])

    return out

# =========================
# Descriptor neighbor search
# =========================

@dataclass
class DescriptorMetricSpec:
    name: str
    sklearn_metric: str
    metric_kwargs: Optional[dict] = None


DESCRIPTOR_METRICS = [
    DescriptorMetricSpec("Cosine", "cosine"),
    DescriptorMetricSpec("Euclidean", "euclidean"),
    DescriptorMetricSpec("Manhattan", "manhattan"),
    DescriptorMetricSpec("Correlation", "correlation"),
    # Mahalanobis requires VI; we will build it after we have X.
]


def topk_neighbors_descriptors(X: np.ndarray, metric_spec: DescriptorMetricSpec, k: int) -> List[np.ndarray]:
    n = X.shape[0]

    nn = NearestNeighbors(
        n_neighbors=min(k + 1, n),
        metric=metric_spec.sklearn_metric,
        metric_params=metric_spec.metric_kwargs
    )
    nn.fit(X)

    dists, idxs = nn.kneighbors(X, return_distance=True)

    out = []
    for i in range(n):
        neigh = idxs[i]
        neigh = neigh[neigh != i]  # drop self
        out.append(neigh[:k])
    return out

# =========================
# Scoring
# =========================

@dataclass
class PerQueryPrecomp:
    total_relevant: int
    ideal_rels_topk: Dict[int, List[float]]  # key: K -> list of top-K ideal graded rels


def precompute_query_relevance(
    cat_sets: List[List[int]],
    k_list: List[int]
) -> List[PerQueryPrecomp]:
    """
    For each query i:
    - total_relevant: count of j != i with Jaccard>0
    - ideal_rels_topk[K]: top-K graded relevance values among all j != i (sorted desc)
    """
    n = len(cat_sets)
    out: List[PerQueryPrecomp] = []
    max_k = max(k_list)

    for i in range(n):
        rels_all = []
        total_rel = 0
        si = cat_sets[i]
        for j in range(n):
            if j == i:
                continue
            r = jaccard_set(si, cat_sets[j])
            if r > 0:
                total_rel += 1
            rels_all.append(r)

        # top max_k ideal rels:
        rels_all = np.array(rels_all, dtype=float)  # length n-1
        if max_k < len(rels_all):
            top_idx = np.argpartition(rels_all, -max_k)[-max_k:]
            top_vals = rels_all[top_idx]
            top_vals = np.sort(top_vals)[::-1]
        else:
            top_vals = np.sort(rels_all)[::-1]

        ideal_map = {K: top_vals[:K].tolist() for K in k_list}
        out.append(PerQueryPrecomp(total_relevant=total_rel, ideal_rels_topk=ideal_map))

    return out


def compute_metrics_for_neighbors(
    neighbors: List[np.ndarray],
    cat_sets: List[List[int]],
    precomp: List[PerQueryPrecomp],
    k: int
) -> Dict[str, float]:
    """
    Compute micro-average metrics over all queries:
    Precision@K, Recall@K, MAP@K, nDCG@K
    """
    n = len(neighbors)
    precs, recs, aps, ndcgs = [], [], [], []

    for i in range(n):
        idx = neighbors[i][:k]
        rels = [jaccard_set(cat_sets[i], cat_sets[j]) for j in idx]
        binrels = [1 if r > 0 else 0 for r in rels]

        prec = float(np.mean(binrels)) if k > 0 else 0.0
        tot_rel = precomp[i].total_relevant
        rec = (float(np.sum(binrels)) / tot_rel) if tot_rel > 0 else 0.0
        ap = average_precision_at_k(binrels, tot_rel, k)
        nd = ndcg_at_k(rels, precomp[i].ideal_rels_topk[k], k)

        precs.append(prec)
        recs.append(rec)
        aps.append(ap)
        ndcgs.append(nd)

    return {
        "Precision@K": float(np.mean(precs)),
        "Recall@K": float(np.mean(recs)),
        "MAP@K": float(np.mean(aps)),
        "nDCG@K": float(np.mean(ndcgs)),
    }


def compute_macro_over_categories(
    neighbors: List[np.ndarray],
    cat_sets: List[List[int]],
    precomp: List[PerQueryPrecomp],
    k: int,
    n_categories: int
) -> Dict[str, float]:
    """
    Macro-average: for each category c, average metrics over queries that contain c, then average over categories.
    """
    # Build category -> query indices
    cat_to_queries: List[List[int]] = [[] for _ in range(n_categories)]
    for i, cs in enumerate(cat_sets):
        for c in cs:
            cat_to_queries[c].append(i)

    per_cat_vals = {"Precision@K": [], "Recall@K": [], "MAP@K": [], "nDCG@K": []}

    for c in range(n_categories):
        qs = cat_to_queries[c]
        if not qs:
            continue

        precs, recs, aps, ndcgs = [], [], [], []
        for i in qs:
            idx = neighbors[i][:k]
            rels = [jaccard_set(cat_sets[i], cat_sets[j]) for j in idx]
            binrels = [1 if r > 0 else 0 for r in rels]

            prec = float(np.mean(binrels)) if k > 0 else 0.0
            tot_rel = precomp[i].total_relevant
            rec = (float(np.sum(binrels)) / tot_rel) if tot_rel > 0 else 0.0
            ap = average_precision_at_k(binrels, tot_rel, k)
            nd = ndcg_at_k(rels, precomp[i].ideal_rels_topk[k], k)

            precs.append(prec)
            recs.append(rec)
            aps.append(ap)
            ndcgs.append(nd)

        per_cat_vals["Precision@K"].append(float(np.mean(precs)))
        per_cat_vals["Recall@K"].append(float(np.mean(recs)))
        per_cat_vals["MAP@K"].append(float(np.mean(aps)))
        per_cat_vals["nDCG@K"].append(float(np.mean(ndcgs)))

    return {k_: float(np.mean(v)) if v else 0.0 for k_, v in per_cat_vals.items()}


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray], float],
    n: int,
    B: int,
    seed: int
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        sample = rng.integers(0, n, size=n)
        vals.append(metric_fn(sample))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# =========================
# Main
# =========================

def main():
    warnings.filterwarnings("ignore")

    df = pd.read_csv(CSV_PATH, sep=SEP)
    if SMILES_COL not in df.columns or CATEGORIES_COL not in df.columns:
        raise ValueError(f"CSV must contain columns '{SMILES_COL}' and '{CATEGORIES_COL}'")

    # Parse + normalize
    df["category_raw"] = df[CATEGORIES_COL].apply(parse_categories)
    df["mol_raw"] = df[SMILES_COL].apply(mol_from_smiles)
    df = df[df["mol_raw"].notnull()].copy()

    df["mol"] = df["mol_raw"].apply(normalize_mol)
    df = df[df["mol"].notnull()].copy()

    # Deduplicate (optional)
    if DEDUP_CANONICAL_SMILES:
        df["can_smiles"] = df["mol"].apply(canonical_smiles)
        df = df.drop_duplicates(subset=["can_smiles"]).copy()

    df = df.reset_index(drop=True)
    mols = df["mol"].tolist()
    n = len(mols)
    print(f"Loaded {n} unique normalized molecules")

    if n == 0:
        raise RuntimeError(
            "No molecules left after parsing/normalization/dedup. "
            "Try disabling normalization steps (REMOVE_SALTS/KEEP_LARGEST_FRAGMENT/NEUTRALIZE) "
            "or inspect invalid SMILES/cases."
        )

    # Build category id mapping
    all_cats = sorted({c for lst in df["category_raw"] for c in lst})
    cat_to_id = {c: i for i, c in enumerate(all_cats)}
    n_cats = len(all_cats)

    cat_sets = [[cat_to_id[c] for c in lst if c in cat_to_id] for lst in df["category_raw"]]

    # Precompute ideal relevance per query for nDCG + total relevant for recall/AP
    print("Precomputing category relevance (ideal rankings for nDCG)...")
    precomp = precompute_query_relevance(cat_sets, TOP_K_LIST)

    # Build descriptors and descriptor NN metrics (including Mahalanobis)
    print("Building descriptor matrix...")
    X = build_descriptor_matrix(mols)

    # Mahalanobis VI
    cov = np.cov(X, rowvar=False)
    # regularize to avoid singularity
    cov += 1e-6 * np.eye(cov.shape[0])
    VI = np.linalg.inv(cov)
    desc_metrics = DESCRIPTOR_METRICS + [DescriptorMetricSpec("Mahalanobis", "mahalanobis", {"VI": VI})]

    results = []

    # ---------- Descriptor retrieval ----------
    for k in TOP_K_LIST:
        for ms in desc_metrics:
            neigh = topk_neighbors_descriptors(X, ms, k=k)

            micro = compute_metrics_for_neighbors(neigh, cat_sets, precomp, k=k)
            macro = compute_macro_over_categories(neigh, cat_sets, precomp, k=k, n_categories=n_cats)

            # Bootstrap CIs (micro only, for speed)
            def micro_metric_from_sample(sample_idx: np.ndarray) -> float:
                # Recompute micro MAP@K on sampled queries only (as representative).
                aps = []
                for i in sample_idx:
                    idx = neigh[i][:k]
                    rels = [jaccard_set(cat_sets[i], cat_sets[j]) for j in idx]
                    binrels = [1 if r > 0 else 0 for r in rels]
                    aps.append(average_precision_at_k(binrels, precomp[i].total_relevant, k))
                return float(np.mean(aps))

            lo, hi = bootstrap_ci(micro_metric_from_sample, n=n, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)

            results.append({
                "representation": "Descriptors",
                "rep_params": f"p={X.shape[1]}",
                "metric": ms.name,
                "K": k,
                **{f"micro_{k_}": v for k_, v in micro.items()},
                **{f"macro_{k_}": v for k_, v in macro.items()},
                "micro_MAP@K_CI_low": lo,
                "micro_MAP@K_CI_high": hi,
            })

    # ---------- Fingerprint retrieval ----------
    fp_builders = []

    # Morgan bit & feature morgan bit
    for r in MORGAN_RADIUS_LIST:
        for b in MORGAN_BITS_LIST:
            fp_builders.append(("MorganBit", f"r={r},b={b}", lambda m, rr=r, bb=b: fp_morgan_bit(m, rr, bb, use_features=False), "bit"))
            fp_builders.append(("FeatMorganBit", f"r={r},b={b}", lambda m, rr=r, bb=b: fp_morgan_bit(m, rr, bb, use_features=True), "bit"))

    # Morgan counts & feature morgan counts
    for r in MORGAN_RADIUS_LIST:
        fp_builders.append(("MorganCount", f"r={r}", lambda m, rr=r: fp_morgan_count(m, rr, use_features=False), "count"))
        fp_builders.append(("FeatMorganCount", f"r={r}", lambda m, rr=r: fp_morgan_count(m, rr, use_features=True), "count"))

    # Other common FPs
    fp_builders += [
        ("RDKBit", "-", fp_rdk_bit, "bit"),
        ("MACCS", "-", fp_maccs, "bit"),
        ("AtomPairCount", "hashed", fp_atompair, "count"),
        ("TopTorsionCount", "hashed", fp_toptorsion, "count"),
    ]

    for fp_name, fp_params, fp_fn, fp_kind in fp_builders:
        print(f"Building fingerprints: {fp_name} ({fp_params}) ...")
        fps = [fp_fn(m) for m in mols]

        sim_specs = get_similarity_specs_for_fp_kind(fp_kind)
        for sim_spec in sim_specs:
            # TODO Fix these
            if fp_name in ("MorganCount", "FeatMorganCount", "AtomPairCount", "TopTorsionCount") and sim_spec.name == "Cosine": continue
            if fp_name in ("MorganCount", "FeatMorganCount", "AtomPairCount", "TopTorsionCount") and sim_spec.name == "Kulczynski": continue
            if fp_name in ("MorganCount", "FeatMorganCount", "AtomPairCount", "TopTorsionCount") and sim_spec.name == "Asymmetric": continue
            if fp_name in ("MorganCount", "FeatMorganCount", "AtomPairCount", "TopTorsionCount") and sim_spec.name == "BraunBlanquet": continue
            if fp_name in ("MorganCount", "FeatMorganCount", "AtomPairCount", "TopTorsionCount") and sim_spec.name == "Sokal": continue
            if fp_name in ("MorganCount", "FeatMorganCount", "AtomPairCount", "TopTorsionCount") and sim_spec.name == "McConnaughey": continue
            if fp_name in ("MorganCount", "FeatMorganCount", "AtomPairCount", "TopTorsionCount") and sim_spec.name == "RogotGoldberg": continue
            if fp_name in ("MorganCount", "FeatMorganCount", "AtomPairCount", "TopTorsionCount") and sim_spec.name == "Russel": continue
            if fp_name in ("MorganCount", "FeatMorganCount", "AtomPairCount", "TopTorsionCount") and sim_spec.name == "OnBit": continue
            for k in TOP_K_LIST:
                print(f"  Retrieving: {fp_name} | {sim_spec.name} | K={k}")
                neigh = topk_neighbors_by_similarity(fps, sim_spec, k=k, exclude_self=True)

                micro = compute_metrics_for_neighbors(neigh, cat_sets, precomp, k=k)
                macro = compute_macro_over_categories(neigh, cat_sets, precomp, k=k, n_categories=n_cats)

                # Bootstrap CIs (micro MAP@K)
                def micro_metric_from_sample(sample_idx: np.ndarray) -> float:
                    aps = []
                    for i in sample_idx:
                        idx = neigh[i][:k]
                        rels = [jaccard_set(cat_sets[i], cat_sets[j]) for j in idx]
                        binrels = [1 if r > 0 else 0 for r in rels]
                        aps.append(average_precision_at_k(binrels, precomp[i].total_relevant, k))
                    return float(np.mean(aps))

                lo, hi = bootstrap_ci(micro_metric_from_sample, n=n, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)

                results.append({
                    "representation": fp_name,
                    "rep_params": fp_params,
                    "metric": sim_spec.name,
                    "K": k,
                    **{f"micro_{k_}": v for k_, v in micro.items()},
                    **{f"macro_{k_}": v for k_, v in macro.items()},
                    "micro_MAP@K_CI_low": lo,
                    "micro_MAP@K_CI_high": hi,
                })

    # ---------- Random baseline ----------
    if RANDOM_BASELINE:
        rng = np.random.default_rng(123)
        for k in TOP_K_LIST:
            neigh = []
            for i in range(n):
                # sample without replacement from all except i
                pool = np.arange(n)
                pool = pool[pool != i]
                idx = rng.choice(pool, size=min(k, n-1), replace=False)
                neigh.append(idx)

            micro = compute_metrics_for_neighbors(neigh, cat_sets, precomp, k=k)
            macro = compute_macro_over_categories(neigh, cat_sets, precomp, k=k, n_categories=n_cats)

            results.append({
                "representation": "Random",
                "rep_params": "-",
                "metric": "Random",
                "K": k,
                **{f"micro_{k_}": v for k_, v in micro.items()},
                **{f"macro_{k_}": v for k_, v in macro.items()},
                "micro_MAP@K_CI_low": np.nan,
                "micro_MAP@K_CI_high": np.nan,
            })

    # Save results
    res_df = pd.DataFrame(results)
    # Sort by micro_MAP@K at a chosen K (e.g., 10) then nDCG
    sort_k = 10 if 10 in TOP_K_LIST else TOP_K_LIST[0]
    res_df["sort_key"] = np.where(res_df["K"] == sort_k, res_df["micro_MAP@K"], -1.0)
    res_df = res_df.sort_values(["sort_key", "K", "micro_nDCG@K"], ascending=[False, True, False]).drop(columns=["sort_key"])

    res_df.to_csv(OUT_TABLE, index=False)
    print(f"\nSaved results to {OUT_TABLE}")

    # Print a compact leaderboard for K=10 (or first K)
    k_show = sort_k
    leader = res_df[res_df["K"] == k_show].copy()
    cols = ["representation", "rep_params", "metric", "K",
            "micro_MAP@K", "micro_MAP@K_CI_low", "micro_MAP@K_CI_high",
            "micro_nDCG@K", "micro_Precision@K", "micro_Recall@K",
            "macro_MAP@K", "macro_nDCG@K"]
    print("\nLeaderboard (higher is better):")
    print(leader[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
