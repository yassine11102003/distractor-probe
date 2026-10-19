"""Étape 03 : combien d'information de POSITION les tokens de patch contiennent-ils ? (distracteur n°1)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.analysis import (linear_probe, position_means, position_variance_share,
                          sample_tokens, split_images)
from src.data import load_features

# Pourquoi le mean-pooling (Pool-Align) élimine la composante de position additive :
# si g_ij = c_ij + mu_ij (contenu + position), alors moyenne(g) = moyenne(c) + moyenne(mu).
# moyenne(mu) est le MÊME vecteur pour toutes les images : un décalage constant qui ne
# distingue aucune image. Retirer mu_ij puis moyenner = moyenner puis soustraire une constante.
# VF, qui aligne chaque position séparément, recopie au contraire mu_ij dans le latent.


def run_probes(X_train, pos_train, X_test, pos_test):
    """Deux sondes : prédire la ligne (pos // 16) et la colonne (pos % 16) du token."""
    g = config.GRID_SIZE
    return {
        "row": linear_probe(X_train, pos_train // g, X_test, pos_test // g,
                            config.SEED, config.PROBE_MAX_ITER),
        "col": linear_probe(X_train, pos_train % g, X_test, pos_test % g,
                            config.SEED, config.PROBE_MAX_ITER),
    }


def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n_images = len(np.load(config.FEATURES_DIR / "labels.npy"))
    train_ids, test_ids = split_images(n_images, config.TEST_FRACTION, config.SEED)
    chance = 1 / config.GRID_SIZE
    results = {"chance": chance, "n_train_images": len(train_ids),
               "n_test_images": len(test_ids), "models": {}}

    for tag in config.MODELS:
        print(f"[{tag}] ...")
        tokens = load_features(tag, "patch_norm")   # tokens normalisés : ceux que VF aligne

        # Composante de position : moyenne de chaque position, calculée sur le TRAIN seulement
        mu = position_means(tokens, train_ids)
        share = position_variance_share(tokens, train_ids, mu)

        # Mêmes tokens tirés pour les deux conditions (graine fixe)
        rng = np.random.default_rng(config.SEED)
        X_train, pos_train = sample_tokens(tokens, train_ids, config.PROBE_TOKENS_PER_IMAGE, rng)
        X_test, pos_test = sample_tokens(tokens, test_ids, config.PROBE_TOKENS_PER_IMAGE, rng)

        raw = run_probes(X_train, pos_train, X_test, pos_test)
        # Intervention : on retire à chaque token la moyenne de SA position (mu vient du train)
        removed = run_probes(X_train - mu[pos_train], pos_train, X_test - mu[pos_test], pos_test)

        results["models"][tag] = {"position_variance_share": round(share, 4),
                                  "accuracy": {"raw": raw, "position_removed": removed}}

    (config.RESULTS_DIR / "03_position_probe.json").write_text(json.dumps(results, indent=2))

    print("\n=== Résumé 03 : accuracy de la sonde de position (hasard = 6.25 %) ===")
    print(f"{'modèle':12s} {'condition':18s} {'ligne':>8s} {'colonne':>8s}")
    for tag, res in results["models"].items():
        for cond, acc in res["accuracy"].items():
            print(f"{tag:12s} {cond:18s} {100 * acc['row']:7.1f}% {100 * acc['col']:7.1f}%")
    print("\nPart de la variance des tokens expliquée par la position seule :")
    for tag, res in results["models"].items():
        print(f"  {tag:12s} {100 * res['position_variance_share']:.1f} %")


if __name__ == "__main__":
    main()
