"streamlit run streamlit_app.py"

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st


DEFAULT_CONFIG_PATH = "config/config.json"
GUI_CONFIG_OUT = "config/gui_last_config.json"


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def parse_seeds(text: str) -> List[int]:
    """
    Accepts:
      - "1,2,3"
      - "1-5"
      - "1,2,5-8"
    """
    text = text.strip()
    if not text:
        return []
    parts = [p.strip() for p in text.split(",")]
    out: List[int] = []
    for p in parts:
        m = re.match(r"^(\d+)\s*-\s*(\d+)$", p)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a <= b:
                out.extend(list(range(a, b + 1)))
            else:
                out.extend(list(range(a, b - 1, -1)))
        else:
            out.append(int(p))

    # unique preserve order
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def run_cli(config_path: str, mode: str) -> str:
    """
    Runs the CLI as a subprocess and streams output into Streamlit.
    Fixes import issues by:
      - running as module: python -m image_clustering.cli
      - setting PYTHONPATH to include ./src
    """
    cmd = [
        sys.executable,
        "-m",
        "image_clustering.cli",
        "--config",
        config_path,
        "--mode",
        mode,
    ]

    st.code(" ".join(cmd), language="bash")

    output_lines: List[str] = []
    log_box = st.empty()

    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(Path.cwd()),
        env=env,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        output_lines.append(line.rstrip("\n"))
        tail = "\n".join(output_lines[-200:])
        log_box.code(tail)

    rc = proc.wait()
    output = "\n".join(output_lines)

    if rc == 0:
        st.success("Finished successfully.")
    else:
        st.error(f"Process failed with exit code {rc}.")

    return output


def main():
    st.set_page_config(page_title="Image Clustering GUI", layout="wide")
    st.title("Image Clustering GUI (local)")

    # Load base config
    base_path = st.sidebar.text_input("Base config path", DEFAULT_CONFIG_PATH)
    if not Path(base_path).exists():
        st.sidebar.error("Base config not found.")
        st.stop()

    cfg = load_json(base_path)

    # --- Sidebar: mode + paths ---
    mode = st.sidebar.selectbox("Mode", ["run", "grid", "stability"], index=0)

    st.sidebar.subheader("Paths")
    input_dir = st.sidebar.text_input("Input images folder", cfg["paths"]["input_dir"])
    output_root = st.sidebar.text_input("Output root", cfg["paths"]["output_root"])
    cache_root = st.sidebar.text_input("Cache root", cfg["paths"]["cache_root"])

    # --- Embeddings ---
    st.sidebar.subheader("Embeddings (DINOv2)")
    model_id_current = cfg.get("embedding", {}).get("dinov2", {}).get("model_id", "facebook/dinov2-base")
    model_id = st.sidebar.selectbox(
        "Model ID",
        ["facebook/dinov2-small", "facebook/dinov2-base", "facebook/dinov2-large"],
        index=["facebook/dinov2-small", "facebook/dinov2-base", "facebook/dinov2-large"].index(model_id_current),
    )
    batch_size = st.sidebar.number_input(
        "Batch size",
        min_value=1,
        max_value=512,
        value=int(cfg.get("embedding", {}).get("batch_size", 32)),
    )
    l2_norm = st.sidebar.checkbox(
        "L2-normalise embeddings",
        value=bool(cfg.get("embedding", {}).get("store", {}).get("l2_normalize", True)),
    )

    # --- Clustering space ---
    st.sidebar.subheader("Clustering space")
    clustering_space_current = cfg.get("clustering", {}).get("space", "umap")
    clustering_space = st.sidebar.selectbox(
        "Cluster in",
        ["umap", "embedding"],
        index=0 if clustering_space_current == "umap" else 1,
    )

    # --- UMAP ---
    st.sidebar.subheader("UMAP")
    umap_n_components_current = int(cfg.get("umap", {}).get("n_components", 2))
    umap_n_components = st.sidebar.selectbox("n_components", [2, 3], index=0 if umap_n_components_current == 2 else 1)
    umap_n_neighbors = st.sidebar.slider("n_neighbors", 5, 200, int(cfg.get("umap", {}).get("n_neighbors", 30)))
    umap_min_dist = st.sidebar.slider("min_dist", 0.0, 1.0, float(cfg.get("umap", {}).get("min_dist", 0.1)), step=0.01)
    umap_metric_current = cfg.get("umap", {}).get("metric", "cosine")
    umap_metric = st.sidebar.selectbox("metric", ["cosine", "euclidean"], index=0 if umap_metric_current == "cosine" else 1)
    umap_seed = st.sidebar.number_input("random_state", min_value=0, max_value=10_000_000, value=int(cfg.get("umap", {}).get("random_state", 42)))

    # --- HDBSCAN ---
    st.sidebar.subheader("HDBSCAN")
    mcs = st.sidebar.slider("min_cluster_size", 2, 500, int(cfg.get("hdbscan", {}).get("min_cluster_size", 10)))
    ms = st.sidebar.slider("min_samples", 1, 200, int(cfg.get("hdbscan", {}).get("min_samples", 10)))
    eps = st.sidebar.slider("cluster_selection_epsilon", 0.0, 5.0, float(cfg.get("hdbscan", {}).get("cluster_selection_epsilon", 0.0)), step=0.01)
    h_metric_current = cfg.get("hdbscan", {}).get("metric", "euclidean")
    h_metric = st.sidebar.selectbox("hdbscan metric", ["euclidean", "manhattan", "cosine"], index=["euclidean", "manhattan", "cosine"].index(h_metric_current))

    # --- Stability ---
    st.sidebar.subheader("Stability (ARI)")
    seeds_text_default = ",".join(map(str, cfg.get("stability", {}).get("seeds", [1, 2, 3, 4, 5])))
    seeds_text = st.sidebar.text_input("Seeds (e.g., 1-10 or 1,2,5-8)", seeds_text_default)
    exclude_noise_for_ari = st.sidebar.checkbox(
        "Exclude noise for ARI",
        value=bool(cfg.get("stability", {}).get("exclude_noise_for_ari", True)),
    )

    # --- Grid search ranges ---
    st.sidebar.subheader("Grid search (ranges)")
    grid_nn = st.sidebar.text_input(
        "UMAP n_neighbors list",
        ",".join(map(str, cfg.get("grid_search", {}).get("umap", {}).get("n_neighbors", [15, 30, 50]))),
    )
    grid_md = st.sidebar.text_input(
        "UMAP min_dist list",
        ",".join(map(str, cfg.get("grid_search", {}).get("umap", {}).get("min_dist", [0.0, 0.1]))),
    )
    grid_mcs = st.sidebar.text_input(
        "HDBSCAN min_cluster_size list",
        ",".join(map(str, cfg.get("grid_search", {}).get("hdbscan", {}).get("min_cluster_size", [10, 20, 30]))),
    )
    grid_ms = st.sidebar.text_input(
        "HDBSCAN min_samples list",
        ",".join(map(str, cfg.get("grid_search", {}).get("hdbscan", {}).get("min_samples", [5, 10]))),
    )

    # --- Build updated config ---
    cfg2 = json.loads(json.dumps(cfg))  # deep copy via JSON roundtrip

    cfg2["paths"]["input_dir"] = input_dir
    cfg2["paths"]["output_root"] = output_root
    cfg2["paths"]["cache_root"] = cache_root

    cfg2["embedding"]["method"] = "dinov2"
    cfg2["embedding"]["dinov2"]["model_id"] = model_id
    cfg2["embedding"]["batch_size"] = int(batch_size)
    cfg2["embedding"]["store"]["l2_normalize"] = bool(l2_norm)

    cfg2["clustering"]["space"] = clustering_space

    cfg2["umap"]["n_components"] = int(umap_n_components)
    cfg2["umap"]["n_neighbors"] = int(umap_n_neighbors)
    cfg2["umap"]["min_dist"] = float(umap_min_dist)
    cfg2["umap"]["metric"] = umap_metric
    cfg2["umap"]["random_state"] = int(umap_seed)

    cfg2["hdbscan"]["min_cluster_size"] = int(mcs)
    cfg2["hdbscan"]["min_samples"] = int(ms)
    cfg2["hdbscan"]["cluster_selection_epsilon"] = float(eps)
    cfg2["hdbscan"]["metric"] = h_metric

    cfg2.setdefault("stability", {})
    cfg2["stability"]["seeds"] = parse_seeds(seeds_text)
    cfg2["stability"]["exclude_noise_for_ari"] = bool(exclude_noise_for_ari)

    cfg2.setdefault("grid_search", {})
    cfg2["grid_search"].setdefault("umap", {})
    cfg2["grid_search"].setdefault("hdbscan", {})
    cfg2["grid_search"]["umap"]["n_neighbors"] = _parse_int_list(grid_nn)
    cfg2["grid_search"]["umap"]["min_dist"] = _parse_float_list(grid_md)
    cfg2["grid_search"]["hdbscan"]["min_cluster_size"] = _parse_int_list(grid_mcs)
    cfg2["grid_search"]["hdbscan"]["min_samples"] = _parse_int_list(grid_ms)

    cfg2.setdefault("export", {})
    if not cfg2["export"].get("date_stamp"):
        cfg2["export"]["date_stamp"] = date.today().strftime("%d-%B-%Y")

    # --- Main area ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Config preview (GUI-generated)")
        st.json(cfg2)

        st.write("This GUI writes a reproducible config snapshot:")
        st.code(GUI_CONFIG_OUT)

        if st.button("Save GUI config snapshot"):
            save_json(GUI_CONFIG_OUT, cfg2)
            st.success(f"Saved: {GUI_CONFIG_OUT}")

    with col2:
        st.subheader("Run")
        st.write("Runs the existing CLI locally via subprocess.")

        if st.button(f"Run mode = {mode}"):
            save_json(GUI_CONFIG_OUT, cfg2)
            output = run_cli(GUI_CONFIG_OUT, mode)

            st.subheader("Post-run hints")
            st.write("Output root:")
            st.code(cfg2["paths"]["output_root"])

            if mode == "grid":
                st.write("Grid search produces: outputs/grid_search_<date_stamp>.csv")
            elif mode == "stability":
                st.write("Stability produces: outputs/stability_<date_stamp>_<embed_id>/")
            else:
                st.write("Run produces a folder with viz_scatter.html and montages/")

            with st.expander("Raw log (full)"):
                st.code(output)


if __name__ == "__main__":
    main()