"""Étape 01 : extraction des features des deux DINOv2, faite une fois puis mise en cache."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Permet de lancer le script depuis la racine : `python scripts/01_extract_features.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.data import download_imagenette, get_loader, select_images
from src.models import extract_features, load_model

KINDS = ("patch_norm", "patch_prenorm", "cls")
RESULTS_FILE = config.RESULTS_DIR / "01_extract_features.json"


def cache_files(tag):
    """Chemins des fichiers .npy d'un modèle, ex. outputs/features/dinov2_cls.npy."""
    return {kind: config.FEATURES_DIR / f"{tag}_{kind}.npy" for kind in KINDS}


def cache_is_valid(files, n_images):
    """Valide si tous les fichiers existent et contiennent le bon nombre d'images."""
    if not all(path.exists() for path in files.values()):
        return False
    return np.load(files["cls"], mmap_mode="r").shape[0] == n_images


def main():
    torch.manual_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Données : les mêmes images à chaque lancement (graine fixe)
    download_imagenette()
    paths, labels = select_images()
    np.save(config.FEATURES_DIR / "labels.npy", np.array(labels))
    (config.FEATURES_DIR / "image_paths.json").write_text(json.dumps(paths, indent=1))
    loader = get_loader(paths, labels)
    print(f"{len(paths)} images sélectionnées, device = {device}")

    # 2) Extraction modèle par modèle (on garde les infos d'un lancement précédent)
    summary = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else {"models": {}}
    summary["n_images"] = len(paths)

    for tag, hub_name in config.MODELS.items():
        files = cache_files(tag)
        if cache_is_valid(files, len(paths)):
            print(f"[{tag}] cache trouvé, rien à refaire.")
            continue

        print(f"[{tag}] chargement de {hub_name} ...")
        model = load_model(hub_name, device)
        t0 = time.time()
        feats = extract_features(model, loader, device)
        for kind, array in feats.items():
            np.save(files[kind], array)

        summary["models"][tag] = {
            "hub_name": hub_name,
            "n_register_tokens": model.num_register_tokens,
            "shapes": {kind: list(array.shape) for kind, array in feats.items()},
            "seconds": round(time.time() - t0, 1),
        }
        del model, feats                  # libère la mémoire avant le modèle suivant
        if device == "cuda":
            torch.cuda.empty_cache()

    RESULTS_FILE.write_text(json.dumps(summary, indent=2))

    # 3) Résumé
    print("\n=== Résumé 01 ===")
    for tag, info in summary["models"].items():
        print(f"{tag:12s} registres={info['n_register_tokens']}  "
              f"shapes={info['shapes']}  ({info['seconds']} s)")
    print(f"Features dans {config.FEATURES_DIR}")


if __name__ == "__main__":
    main()
