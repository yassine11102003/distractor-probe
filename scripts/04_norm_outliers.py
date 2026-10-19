"""Étape 04 : tokens d'artefact à forte norme (Darcet et al., 2024) et leur effet sur le pooling (distracteur n°2)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.analysis import mean_pool, outlier_mask, rowwise_cosine
from src.data import load_display_image, load_features
from src.plots import plot_cosine_histograms, plot_norm_histograms, plot_outlier_overlay


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    g = config.GRID_SIZE

    norms_by_model, thresholds, masks_by_model, cosines_by_model = {}, {}, {}, {}
    results = {"rule": f"norme > médiane + {config.OUTLIER_IQR_K} x IQR", "models": {}}

    for tag in config.MODELS:
        print(f"[{tag}] ...")
        # Normes calculées AVANT la LayerNorm finale : après elle, tous les tokens ont la même échelle
        mask, norms, threshold = outlier_mask(load_features(tag, "patch_prenorm"), config.OUTLIER_IQR_K)

        # Impact sur le pooling : moyenne des tokens NORMALISÉS (ceux que Pool-Align moyenne)
        tokens = load_features(tag, "patch_norm")
        cosines = rowwise_cosine(mean_pool(tokens), mean_pool(tokens, mask))
        affected = mask.any(axis=1)

        norms_by_model[tag] = norms
        thresholds[tag] = threshold
        masks_by_model[tag] = [mask[i].reshape(g, g) for i in config.VIS_IMAGE_INDICES]
        cosines_by_model[tag] = cosines[affected]
        results["models"][tag] = {
            "median_norm": round(float(np.median(norms)), 2),
            "threshold": round(threshold, 2),
            "max_norm": round(float(norms.max()), 2),
            "pct_outlier_tokens": round(100 * float(mask.mean()), 3),
            "pct_images_with_outlier": round(100 * float(affected.mean()), 2),
            "mean_outliers_per_affected_image": round(float(mask[affected].sum(axis=1).mean()), 2)
                                                if affected.any() else 0.0,
            "pool_cosine_mean_all_images": round(float(cosines.mean()), 5),
            "pool_cosine_mean_affected_images": round(float(cosines[affected].mean()), 5)
                                                if affected.any() else None,
            "pool_cosine_min": round(float(cosines.min()), 5),
        }

    # Figures
    paths = json.loads((config.FEATURES_DIR / "image_paths.json").read_text())
    labels = np.load(config.FEATURES_DIR / "labels.npy")
    class_names = [config.CLASS_NAMES[wnid] for wnid in sorted(config.CLASS_NAMES)]
    images = [load_display_image(paths[i]) for i in config.VIS_IMAGE_INDICES]
    titles = [class_names[labels[i]] for i in config.VIS_IMAGE_INDICES]
    plot_norm_histograms(norms_by_model, thresholds, config.FIGURES_DIR / "04_norm_hist.png")
    plot_outlier_overlay(images, masks_by_model, titles, config.PATCH_SIZE,
                         config.FIGURES_DIR / "04_outliers_overlay.png")
    plot_cosine_histograms(cosines_by_model, config.FIGURES_DIR / "04_pool_cosine.png")

    (config.RESULTS_DIR / "04_norm_outliers.json").write_text(json.dumps(results, indent=2))

    print(f"\n=== Résumé 04 (outlier : {results['rule']}) ===")
    for tag, r in results["models"].items():
        print(f"{tag:12s} médiane={r['median_norm']:.1f}  seuil={r['threshold']:.1f}  max={r['max_norm']:.1f}")
        print(f"{'':12s} tokens outliers = {r['pct_outlier_tokens']:.2f} %   "
              f"images touchées = {r['pct_images_with_outlier']:.1f} %   "
              f"(en moyenne {r['mean_outliers_per_affected_image']} par image touchée)")
        print(f"{'':12s} cos(pool complet, pool sans outliers) : moyenne = {r['pool_cosine_mean_all_images']}"
              f", images touchées = {r['pool_cosine_mean_affected_images']}, min = {r['pool_cosine_min']}")
    print(f"Figures : {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
