"""Fonctions de figures (matplotlib), sauvegardées en PNG."""
import matplotlib

matplotlib.use("Agg")   # pas besoin d'écran : on écrit directement des fichiers
import matplotlib.pyplot as plt
import numpy as np


def plot_pca_grid(images, pca_maps, titles, out_path):
    """
    Une ligne par image : l'image, puis la carte PCA-RGB de chaque modèle.
    images : liste de (224, 224, 3) ; pca_maps : {nom du modèle: liste de cartes (16, 16, 3)}
    """
    n_rows, n_cols = len(images), 1 + len(pca_maps)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    for row, image in enumerate(images):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(titles[row], fontsize=10)
        for col, (name, maps) in enumerate(pca_maps.items(), start=1):
            # "nearest" : on voit les 16x16 patchs tels quels, sans lissage qui créerait un faux dégradé
            axes[row, col].imshow(maps[row], interpolation="nearest")
            axes[row, col].set_title(name, fontsize=10)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_norm_histograms(norms_by_model, thresholds, out_path):
    """Un histogramme des normes de tokens par modèle (axe y en log), avec le seuil d'outlier."""
    fig, axes = plt.subplots(1, len(norms_by_model), figsize=(5 * len(norms_by_model), 3.5))
    for ax, (name, norms) in zip(axes, norms_by_model.items()):
        ax.hist(norms.ravel(), bins=200, color="tab:blue")
        ax.axvline(thresholds[name], color="tab:red", linestyle="--", label="seuil outlier")
        ax.set_yscale("log")   # log : la petite bosse des artefacts reste visible
        ax.set_title(name)
        ax.set_xlabel("norme du token (avant LayerNorm finale)")
        ax.set_ylabel("nombre de tokens")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_outlier_overlay(images, masks_by_model, titles, patch_size, out_path):
    """
    Une ligne par image, une colonne par modèle : l'image avec ses patchs outliers en rouge.
    masks_by_model : {nom du modèle: liste de masques booléens (16, 16)}
    """
    n_rows, n_cols = len(images), len(masks_by_model)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)
    for row, image in enumerate(images):
        for col, (name, masks) in enumerate(masks_by_model.items()):
            # On agrandit le masque 16x16 en 224x224 : chaque case devient un carré de 14x14 pixels
            big_mask = np.kron(masks[row], np.ones((patch_size, patch_size))).astype(bool)
            overlay = np.zeros((*big_mask.shape, 4))
            overlay[big_mask] = (1, 0, 0, 0.55)          # rouge semi-transparent
            axes[row, col].imshow(image)
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(f"{name} : {titles[row]} ({int(masks[row].sum())} outliers)",
                                     fontsize=9)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_cosine_histograms(cosines_by_model, out_path):
    """
    Distribution du cosinus pool complet / pool sans outliers.
    cosines_by_model : {nom du modèle: cosinus des SEULES images ayant au moins un outlier}
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name, cosines in cosines_by_model.items():
        if len(cosines):
            ax.hist(cosines, bins=50, alpha=0.6, label=f"{name} ({len(cosines)} images)")
    ax.set_xlabel("cos(moyenne de tous les tokens, moyenne sans les outliers)")
    ax.set_ylabel("nombre d'images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
