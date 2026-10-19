"""
Chargement de DINOv2 (torch.hub) et extraction des features.
"""
import numpy as np
import torch

import config


def load_model(hub_name, device):
    """Charge un DINOv2 pré-entraîné depuis torch.hub, en mode évaluation (poids gelés)."""
    model = torch.hub.load(config.HUB_REPO, hub_name)
    return model.eval().to(device)


def print_first_batch(feats):
    """Affiche les clés renvoyées par forward_features et leurs shapes (vérification de l'API)."""
    print(f"  forward_features -> clés : {list(feats.keys())}")
    for key, value in feats.items():
        shape = tuple(value.shape) if torch.is_tensor(value) else value
        print(f"    {key:20s} {shape}")


@torch.no_grad()
def extract_features(model, loader, device):
    """
    Passe toutes les images dans le modèle et renvoie un dict de tableaux numpy float16 :
      - patch_norm    (N, 256, 1024) : tokens de patch APRÈS la LayerNorm finale
      - patch_prenorm (N, 256, 1024) : les mêmes tokens AVANT la LayerNorm finale
      - cls           (N, 1024)      : token CLS après la LayerNorm finale
    """
    n_images = len(loader.dataset)
    n_patches = config.GRID_SIZE ** 2
    n_reg = model.num_register_tokens   # 0 pour dinov2_vitl14, 4 pour dinov2_vitl14_reg
    print(f"  registres : {n_reg}")

    # Tableaux pré-alloués en float16 (2 octets par valeur : ~1 Go par tableau de tokens
    # pour 2000 images), remplis lot par lot.
    out = {
        "patch_norm": np.empty((n_images, n_patches, config.EMBED_DIM), dtype=np.float16),
        "patch_prenorm": np.empty((n_images, n_patches, config.EMBED_DIM), dtype=np.float16),
        "cls": np.empty((n_images, config.EMBED_DIM), dtype=np.float16),
    }

    start = 0
    for batch_idx, (images, _) in enumerate(loader):
        images = images.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            feats = model.forward_features(images)
        if batch_idx == 0:
            print_first_batch(feats)

        # x_prenorm = [CLS, registres (n_reg), 256 patchs], AVANT la LayerNorm finale.
        # Pour étudier les normes (artefacts, script 04), il faut ces tokens-là :
        # la LayerNorm recentre et remet chaque token à la même échelle, ce qui
        # effacerait justement les différences de norme qu'on veut mesurer.
        patch_prenorm = feats["x_prenorm"][:, 1 + n_reg:]
        patch_norm = feats["x_norm_patchtokens"]
        assert patch_prenorm.shape[1:] == (n_patches, config.EMBED_DIM), patch_prenorm.shape
        assert patch_norm.shape[1:] == (n_patches, config.EMBED_DIM), patch_norm.shape

        end = start + images.shape[0]
        out["patch_norm"][start:end] = patch_norm.half().cpu().numpy()
        out["patch_prenorm"][start:end] = patch_prenorm.half().cpu().numpy()
        out["cls"][start:end] = feats["x_norm_clstoken"].half().cpu().numpy()
        start = end
        print(f"  {end}/{n_images} images", end="\r")

    print()
    return out
