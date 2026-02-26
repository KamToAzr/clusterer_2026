# Image Clustering Pipeline (DINOv2 + UMAP + HDBSCAN)

## Overview

This project implements a reproducible, configuration-driven image clustering pipeline for academic image analysis.

Pipeline stages:

1. Deterministic image ingestion (manifest creation)
2. Feature extraction using DINOv2 embeddings
3. Optional dimensionality reduction with UMAP
4. Density-based clustering using HDBSCAN
5. Evaluation (Silhouette, DBCV)
6. Interactive visualisation (Bokeh)
7. Cluster montage generation
8. Run logging for systematic comparison of runs

All major parameters are controlled via `config/config.json`.

---

## Project structure

```text
clusterer_2026/
├── README.md
├── config/
│   └── config.json
├── cache/
│   └── embeddings/
├── outputs/
│   ├── DINOV2_<date>_<run_id>/
│   └── run_log.csv
├── src/
│   └── image_clustering/
│       ├── cli.py
│       ├── io/
│       │   └── manifest.py
│       ├── embeddings/
│       │   └── dinov2.py
│       ├── reduce/
│       │   └── umap_reduce.py
│       ├── cluster/
│       │   ├── hdbscan_cluster.py
│       │   └── evaluate.py
│       ├── viz/
│       │   ├── bokeh_scatter.py
│       │   └── montage.py
│       └── utils/
│           └── hashing.py
└── venv/
```

---

## Pipeline logic

### 1) Manifest

All images in `input_dir` are scanned and indexed deterministically.

The manifest stores:

- `image_id` (stable hash)
- File paths
- Dimensions and basic metadata

This ensures reproducibility across runs.

---

### 2) Embeddings (DINOv2)

Images are converted into feature vectors using a Hugging Face DINOv2 checkpoint (default: `facebook/dinov2-base`).

Output:

- Embedding matrix `X` with shape `(N, D)` (e.g., `D = 768` for `dinov2-base`)

Embeddings are cached in:

```text
cache/embeddings/<embed_id>/
```

Embeddings are recomputed only if:

- Images change
- `dinov2.model_id` changes
- Preprocessing changes (e.g., crop/size)
- Normalisation settings change

---

### 3) Clustering space

Config option:

```json
"clustering": { "space": "embedding" }
```

Valid values:

- `"embedding"` → cluster in the original embedding space `(N × D)`
- `"umap"` → cluster in the UMAP-projected space (2D or 3D)

UMAP is always computed for visualization. It affects clustering only if `space = "umap"`.

---

### 4) Clustering (HDBSCAN)

HDBSCAN performs density-based clustering and labels low-density points as noise.

Outputs include:

- `cluster_label` (noise is `-1`)
- `probability` (membership strength)

Main output file:

- `assignments.csv`  
  (maps each image to its cluster label and coordinates used for plotting)

---

### 5) Evaluation

Evaluation metrics are computed in the same space where clustering is performed (embedding space or UMAP space), and can optionally exclude noise.

Outputs:

- `run_metrics.json`
- `cluster_summary.csv`  
  (cluster sizes and UMAP centroids for labeling)

---

### 6) Visualisation

Each run produces:

- `viz_scatter.html`  
  Interactive scatter plot (UMAP layout) with:
  - Cluster colouring
  - Noise toggle
  - Centroid labels
  - Run metrics displayed beneath the plot

---

### 7) Cluster montages

Each run produces:

```text
montages/
```

Contains:

- Grid images per cluster (random sample per cluster) for manual validation

Optional:

- Noise montage (if enabled in config)

---

### 8) Run log

Each run appends a row to:

```text
outputs/run_log.csv
```

Stored fields typically include:

- Embedding model id
- Clustering space
- UMAP parameters
- HDBSCAN parameters
- `silhouette_filtered`
- `dbcv_filtered`
- `noise_fraction`
- Number of clusters
- Output directory path

---

## Configuration glossary

### `project`

| Key | Meaning |
|---|---|
| `seed` | Seed for reproducibility |
| `device` | `"cpu"`, `"cuda"`, or `"auto"` |

### `paths`

| Key | Meaning |
|---|---|
| `input_dir` | Folder containing images |
| `output_root` | Directory where run folders are written |
| `cache_root` | Embedding cache location |

### `ingest`

| Key | Meaning |
|---|---|
| `extensions` | File extensions to include |
| `max_images` | Optional cap for testing (`null` = no cap) |

### `embedding`

| Key | Meaning |
|---|---|
| `method` | Embedding backend (currently `"dinov2"`) |
| `dinov2.model_id` | Hugging Face DINOv2 checkpoint id |
| `batch_size` | Batch size for embedding inference |
| `preprocess.input_size` | Resize + center crop size |
| `store.l2_normalize` | Whether to L2-normalise embeddings |

**Impact**

- Model and preprocessing affect the representation geometry (and thus clustering).
- Batch size affects speed and memory usage, not clustering results.

### `clustering`

| Key | Meaning |
|---|---|
| `space` | `"embedding"` or `"umap"` |

**Impact**

- If `"embedding"` → UMAP changes only the plot layout.
- If `"umap"` → UMAP also changes the clustering input.

### `umap`

| Key | Meaning |
|---|---|
| `n_components` | 2 or 3 |
| `n_neighbors` | Local neighbourhood size |
| `min_dist` | How tightly points are packed |
| `metric` | Distance metric |
| `random_state` | Seed for reproducibility (if set, disables UMAP parallelism) |

### `hdbscan`

| Key | Meaning |
|---|---|
| `min_cluster_size` | Minimum size for a cluster |
| `min_samples` | Density strictness (higher → more noise) |
| `cluster_selection_epsilon` | Cluster merging tolerance |
| `metric` | Distance metric used by HDBSCAN |

### `evaluation`

| Key | Meaning |
|---|---|
| `exclude_noise` | If true, compute metrics excluding label `-1` |

### `viz`

| Key | Meaning |
|---|---|
| `bokeh.show_noise_default` | Noise visible by default in scatter plot |
| `montage.n_per_cluster` | Images per cluster montage |
| `montage.include_noise` | Generate montage for noise (`-1`) |

### `export`

| Key | Meaning |
|---|---|
| `date_stamp` | Used in output folder naming |

---

## Evaluation metrics

### Silhouette score

Measures cluster cohesion versus separation.

- Range: `[-1, 1]`
- Higher is better
- Computed on non-noise points if `exclude_noise = true`

Interpretation (rough guide):

- > 0.5 → strong separation
- 0.2–0.5 → moderate structure
- < 0.2 → weak structure

### DBCV (Density-Based Cluster Validation)

Designed specifically for density-based clustering.

- Range: `[-1, 1]`
- Higher is better
- Captures within-cluster density versus between-cluster separation
- Often more appropriate than silhouette for HDBSCAN

### Noise fraction

Proportion of images assigned to noise:

```text
n_noise / n_total
```

High noise may indicate strict parameters or weak density structure.

---

## Running the pipeline

From the project root:

```bash
python -m image_clustering.cli --config config/config.json
```

Outputs are written to:

```text
outputs/DINOV2_<date>_<run_id>/
```
## Selecting the “best” clustering run (grid search, using the "--mode grid" parameter)

The grid search outputs one row per parameter combination in `outputs/grid_search_<date>.csv`.
We select a final run using the following decision logic.

### 1) Feasibility filters (exclude degenerate solutions)

Discard runs that fail any of these checks:

- **Degenerate typology**: `n_clusters < 3`
- **Excessive noise** (default threshold; adjust to project goals): `noise_fraction > 0.60`
- **Dominant single cluster** (default threshold): `largest_cluster_share > 0.50`
- **Missing validity metrics**: `dbcv_filtered` is `NA` (and optionally `silhouette_filtered` is `NA`)

These filters remove parameter settings that do not produce a usable cluster structure.

### 2) Primary ranking criterion (density validity)

Among remaining runs, rank by:

1. **DBCV** (`dbcv_filtered`) — descending

DBCV is prioritised because the clustering method is density-based (HDBSCAN).

### 3) Secondary ranking criterion (separation robustness)

Use as tie-breakers (in order):

2. **Silhouette** (`silhouette_filtered`) — descending  
3. **Noise fraction** (`noise_fraction`) — ascending

Silhouette provides a complementary separation check (computed on the filtered, non-noise set if configured).

### 4) Structural sanity checks (typology usefulness)

Prefer runs that satisfy:

- **Balanced cluster sizes**: higher `cluster_entropy`
- **No single cluster dominates**: lower `largest_cluster_share`
- **Pragmatic number of clusters**: `n_clusters` within an analytically useful range  
  (project-dependent; e.g., 15–80 for typology work)

If two runs have similar DBCV/silhouette, choose the one with lower noise and more balanced cluster sizes.

### 5) Final selection via manual validation

Select the top ~3–5 runs after ranking and inspect:

- `viz_scatter.html` (global separation, noise patterns)
- `montages/` (intra-cluster coherence; inter-cluster differentiation)

The final choice is the best trade-off between:
validity metrics (DBCV/silhouette), noise level, cluster balance, and substantive interpretability.

# Execution Modes

The pipeline supports three execution modes:

-   `--mode run`
-   `--mode grid`
-   `--mode stability`

Each mode serves a distinct methodological purpose.

------------------------------------------------------------------------

## 1) `--mode run` --- Standard Clustering

This is the default analysis mode.

### What it does

1.  Loads (or computes) embeddings.
2.  Runs UMAP (for visualisation and optionally clustering).
3.  Runs HDBSCAN in the space defined by:

``` json
"clustering": { "space": "umap" }
```

or

``` json
"clustering": { "space": "embedding" }
```

4.  Computes evaluation metrics:
    -   `silhouette_filtered`
    -   `dbcv_filtered`
    -   `noise_fraction`
    -   `n_clusters`
    -   `largest_cluster_share`
    -   `cluster_entropy`
5.  Writes:
    -   `assignments.csv`
    -   `umap_coords.csv`
    -   `run_metrics.json`
    -   `viz_scatter.html`
    -   cluster montages

### Purpose

Use this mode once optimal parameters have been selected.

------------------------------------------------------------------------

## 2) `--mode grid` --- Systematic Parameter Search

This mode evaluates multiple UMAP and HDBSCAN parameter combinations.

### What it does

-   Reuses cached embeddings.
-   Iterates over parameter ranges defined in:

``` json
"grid_search": {
  "umap": {
    "n_neighbors": [...],
    "min_dist": [...]
  },
  "hdbscan": {
    "min_cluster_size": [...],
    "min_samples": [...]
  }
}
```

-   Uses the clustering space defined in:

``` json
"clustering": { "space": "umap" | "embedding" }
```

-   Computes evaluation metrics for each combination.
-   Writes a single ranked file:

```{=html}
<!-- -->
```
    outputs/grid_search_<date>.csv

No visualisations or montages are produced.

------------------------------------------------------------------------

### Ranking Logic

Each parameter combination receives a ranking score:

rank_score = 0.5 \* DBCV + 0.3 \* Silhouette - 0.2 \* Noise Fraction

#### Interpretation of weights

-   **DBCV (0.5)**\
    Primary criterion.\
    Clustering is density-based (HDBSCAN), so DBCV is weighted highest.

-   **Silhouette (0.3)**\
    Secondary separation metric.\
    Helps avoid selecting solutions that are dense but poorly separated.

-   **Noise Fraction (-0.2)**\
    Penalises solutions that discard too many points as noise.

Missing values are handled conservatively: - Missing DBCV or silhouette
→ treated as `0` - Missing noise fraction → treated as `1` (worst-case
penalty)

The CSV is automatically sorted by: 1. `rank_score` (descending) 2.
`dbcv_filtered` (descending) 3. `silhouette_filtered` (descending) 4.
`noise_fraction` (ascending)

------------------------------------------------------------------------

### Recommended Selection Workflow

1.  Exclude degenerate runs:

    -   `n_clusters < 3`
    -   `noise_fraction > 0.6`
    -   `largest_cluster_share > 0.5`

2.  Inspect top-ranked runs.

3.  Validate the best 3--5 candidates using `--mode run`.

------------------------------------------------------------------------

## 3) `--mode stability` --- Cluster Robustness Analysis

This mode evaluates clustering stability across random seeds.

### What it does

-   Re-runs the selected configuration using different seeds:

``` json
"stability": {
  "seeds": [1, 2, 3, 4, 5],
  "exclude_noise_for_ari": true
}
```

-   Computes pairwise Adjusted Rand Index (ARI).
-   Writes:

```{=html}
<!-- -->
```
    outputs/stability_<date>_<embed_id>/
        ari_matrix.csv
        stability_summary.json

------------------------------------------------------------------------

### Interpretation of ARI

  Mean ARI     Interpretation
  ------------ -------------------
  \> 0.90      Extremely stable
  0.75--0.90   Very stable
  0.60--0.75   Moderately stable
  0.40--0.60   Weak stability
  \< 0.40      Unstable

For research-grade robustness, aim for:

Mean ARI ≥ 0.75

------------------------------------------------------------------------

### Important Notes

-   Stability analysis is meaningful when clustering in UMAP space.
-   If clustering is performed in embedding space, ARI will typically be
    ≈1.0 because embeddings are deterministic.

------------------------------------------------------------------------

# Summary of Modes

  Mode          Purpose                            Produces
  ------------- ---------------------------------- ----------------------
  `run`         Final clustering + visualisation   Full output folder
  `grid`        Parameter optimisation             Ranked CSV
  `stability`   Robustness check (ARI)             ARI matrix + summary
