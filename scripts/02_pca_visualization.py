"""Étape 02 : PCA à 3 composantes des tokens de patch, affichée en RGB (esprit de la Figure 3)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.analysis import pca_rgb
from src.data import load_display_image, load_features
from src.plots import plot_pca_grid


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Les images à montrer, et leur classe pour le titre
    paths = json.loads((config.FEATURES_DIR / "image_paths.json").read_text())
    labels = np.load(config.FEATURES_DIR / "labels.npy")
    class_names = [config.CLASS_NAMES[wnid] for wnid in sorted(config.CLASS_NAMES)]
    ids = config.VIS_IMAGE_INDICES
    images = [load_display_image(paths[i]) for i in ids]
    titles = [class_names[labels[i]] for i in ids]

    # Une PCA par image et par modèle, sur les tokens normalisés (ceux que VF aligne)
    pca_maps = {}
    results = {"images": [paths[i] for i in ids], "explained_variance_top3": {}}
    for tag in config.MODELS:
        tokens = load_features(tag, "patch_norm")
        maps, explained = [], []
        for i in ids:
            rgb, var = pca_rgb(tokens[i], config.GRID_SIZE)
            maps.append(rgb)
            explained.append(round(var, 3))
        pca_maps[tag] = maps
        results["explained_variance_top3"][tag] = explained

    fig_path = config.FIGURES_DIR / "02_pca_rgb.png"
    plot_pca_grid(images, pca_maps, titles, fig_path)
    (config.RESULTS_DIR / "02_pca_visualization.json").write_text(json.dumps(results, indent=2))

    print("=== Résumé 02 ===")
    print("Variance expliquée par les 3 premiers axes (une valeur par image) :")
    for tag, values in results["explained_variance_top3"].items():
        print(f"  {tag:12s} {values}")
    print(f"Figure : {fig_path}")


if __name__ == "__main__":
    main()
