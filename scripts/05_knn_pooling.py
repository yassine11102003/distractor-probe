"""Étape 05 : un simple mean-pool (ce qu'utilise Pool-Align) garde-t-il la sémantique ? k-NN sur 10 classes."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.analysis import knn_accuracy, mean_pool, outlier_mask, split_images
from src.data import load_features


def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = np.load(config.FEATURES_DIR / "labels.npy")
    # Même split par image qu'au script 03 (même graine)
    train_ids, test_ids = split_images(len(labels), config.TEST_FRACTION, config.SEED)
    results = {"k": config.KNN_K, "chance": 1 / len(config.CLASS_NAMES),
               "n_test_images": len(test_ids), "models": {}}

    for tag in config.MODELS:
        print(f"[{tag}] ...")
        tokens = load_features(tag, "patch_norm")
        mask, _, _ = outlier_mask(load_features(tag, "patch_prenorm"), config.OUTLIER_IQR_K)
        descriptors = {
            "mean_pool": mean_pool(tokens),
            "mean_pool_no_outliers": mean_pool(tokens, mask),
            "cls": np.asarray(load_features(tag, "cls"), dtype=np.float32),
        }
        results["models"][tag] = {
            name: round(knn_accuracy(X[train_ids], labels[train_ids],
                                     X[test_ids], labels[test_ids], config.KNN_K), 4)
            for name, X in descriptors.items()
        }

    (config.RESULTS_DIR / "05_knn_pooling.json").write_text(json.dumps(results, indent=2))

    n_test = results["n_test_images"]
    print(f"\n=== Résumé 05 : k-NN (k={config.KNN_K}, cosinus), {n_test} images de test, hasard = 10 % ===")
    print(f"{'modèle':12s} {'descripteur':24s} {'accuracy':>9s} {'erreurs':>8s}")
    for tag, accs in results["models"].items():
        for name, acc in accs.items():
            print(f"{tag:12s} {name:24s} {100 * acc:8.1f}% {round((1 - acc) * n_test):8d}")


if __name__ == "__main__":
    main()
