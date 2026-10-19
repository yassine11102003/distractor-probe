"""Fonctions d'analyse pures : tableaux numpy en entrée, tableaux ou nombres en sortie."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CHUNK = 100   # nombre d'images lues à la fois, pour ne pas charger 1 Go d'un coup


def split_images(n_images, test_fraction, seed):
    """Sépare les INDICES D'IMAGES en train/test : tous les tokens d'une image restent du même côté."""
    perm = np.random.default_rng(seed).permutation(n_images)
    n_test = int(n_images * test_fraction)
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])


def position_means(tokens, image_ids):
    """Moyenne de chaque position (i, j) sur les images données : tableau (256, 1024)."""
    total = np.zeros(tokens.shape[1:], dtype=np.float64)
    for start in range(0, len(image_ids), CHUNK):
        total += tokens[image_ids[start:start + CHUNK]].astype(np.float32).sum(axis=0)
    return (total / len(image_ids)).astype(np.float32)


def position_variance_share(tokens, image_ids, pos_means):
    """
    Part de la variance des tokens expliquée par la seule position (R² d'une ANOVA à un facteur) :
    variance entre les moyennes par position / variance totale des tokens.
    """
    global_mean = pos_means.mean(axis=0)   # = moyenne de tous les tokens (chaque position compte autant)
    between = ((pos_means - global_mean) ** 2).sum(axis=1).mean()
    total, count = 0.0, 0
    for start in range(0, len(image_ids), CHUNK):
        x = tokens[image_ids[start:start + CHUNK]].astype(np.float32)
        total += ((x - global_mean) ** 2).sum()
        count += x.shape[0] * x.shape[1]
    return float(between / (total / count))


def sample_tokens(tokens, image_ids, n_per_image, rng):
    """
    Tire n_per_image tokens au hasard (sans remise) dans chaque image.
    Renvoie X (n_images * n_per_image, 1024) et la position de chaque token (indice 0 à 255).
    """
    n_patches = tokens.shape[1]
    X, positions = [], []
    for i in image_ids:
        p = np.sort(rng.choice(n_patches, n_per_image, replace=False))
        X.append(tokens[i][p].astype(np.float32))
        positions.append(p)
    return np.concatenate(X), np.concatenate(positions)


def token_norms(tokens):
    """Norme L2 de chaque token : (N, 256, 1024) -> (N, 256)."""
    norms = np.empty(tokens.shape[:2], dtype=np.float32)
    for start in range(0, tokens.shape[0], CHUNK):
        x = tokens[start:start + CHUNK].astype(np.float32)
        norms[start:start + CHUNK] = np.linalg.norm(x, axis=-1)
    return norms


def iqr_threshold(values, k):
    """Seuil robuste de Tukey : médiane + k * IQR (IQR = quantile 75 % - quantile 25 %)."""
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return float(median + k * (q75 - q25))


def outlier_mask(prenorm_tokens, k):
    """
    Tokens d'artefact : norme (avant LayerNorm finale) > médiane + k * IQR.
    Renvoie (masque booléen (N, 256), normes (N, 256), seuil).
    """
    norms = token_norms(prenorm_tokens)
    threshold = iqr_threshold(norms, k)
    return norms > threshold, norms, threshold


def mean_pool(tokens, mask=None):
    """Moyenne des tokens de chaque image -> (N, 1024). Si mask (N, 256) est donné, ses tokens sont exclus."""
    pooled = np.empty((tokens.shape[0], tokens.shape[2]), dtype=np.float32)
    for start in range(0, tokens.shape[0], CHUNK):
        x = tokens[start:start + CHUNK].astype(np.float32)     # (n, 256, 1024)
        if mask is None:
            pooled[start:start + CHUNK] = x.mean(axis=1)
        else:
            keep = ~mask[start:start + CHUNK]                   # (n, 256)
            pooled[start:start + CHUNK] = ((x * keep[..., None]).sum(axis=1)
                                           / np.maximum(keep.sum(axis=1), 1)[:, None])
    return pooled


def rowwise_cosine(a, b):
    """Cosinus entre a[i] et b[i] pour chaque ligne i de deux tableaux (N, D)."""
    return (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def knn_accuracy(X_train, y_train, X_test, y_test, k):
    """k-NN, distance cosinus, vote majoritaire des k plus proches voisins. Renvoie l'accuracy de test."""
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(X_train, y_train)
    return float(knn.score(X_test, y_test))


def linear_probe(X_train, y_train, X_test, y_test, seed, max_iter):
    """Sonde linéaire : régression logistique sur features standardisées. Renvoie l'accuracy de test."""
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=max_iter, random_state=seed))
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def pca_rgb(tokens, grid_size):
    """
    PCA à 3 composantes des tokens d'UNE image, remise sur la grille et vue comme du RGB.
    tokens : (256, 1024) -> renvoie (carte (16, 16, 3) dans [0, 1], variance expliquée par les 3 axes)
    """
    x = tokens.astype(np.float32)
    x = x - x.mean(axis=0)                                  # la PCA se fait sur des données centrées
    _, s, vt = np.linalg.svd(x, full_matrices=False)        # lignes de vt = axes principaux
    proj = x @ vt[:3].T                                     # (256, 3) : coordonnées sur les 3 premiers axes
    proj = (proj - proj.min(axis=0)) / (proj.max(axis=0) - proj.min(axis=0) + 1e-8)  # chaque axe -> [0, 1]
    explained = (s[:3] ** 2).sum() / (s ** 2).sum()         # s² = variance portée par chaque axe
    return proj.reshape(grid_size, grid_size, 3), float(explained)
