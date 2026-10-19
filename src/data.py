"""
Données : téléchargement d'Imagenette, sélection équilibrée des images,
Dataset PyTorch et DataLoader.
"""
import random
import tarfile
import urllib.request

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config


def download_imagenette():
    """
    Télécharge et décompresse Imagenette (~330 Mo), avec vérifications.
    Un fichier témoin n'est écrit qu'une fois tout vérifié : un téléchargement
    ou une décompression interrompus ne sont donc jamais pris pour un succès.
    """
    done_marker = config.DATA_DIR / ".imagenette_ok"
    if done_marker.exists():
        return
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = config.DATA_DIR / "imagenette2-320.tgz"

    print(f"Téléchargement de {config.IMAGENETTE_URL} ...")
    _, headers = urllib.request.urlretrieve(config.IMAGENETTE_URL, archive)
    received, expected = archive.stat().st_size, int(headers["Content-Length"])
    if received != expected:
        archive.unlink()
        raise RuntimeError(f"Téléchargement incomplet ({received} / {expected} octets) : relance.")

    print("Décompression ...")
    with tarfile.open(archive) as tar:
        # filter="data" refuse les chemins dangereux (../, liens absolus) dans l'archive
        tar.extractall(config.DATA_DIR, filter="data")

    # tarfile s'arrête SANS erreur sur une archive tronquée : on vérifie chaque classe
    for wnid in config.CLASS_NAMES:
        if not any((config.IMAGENETTE_DIR / "val" / wnid).glob("*")):
            raise RuntimeError(f"Classe {wnid} vide après décompression : supprime {config.DATA_DIR} et relance.")
    archive.unlink()          # l'archive n'est plus utile (libère ~330 Mo)
    done_marker.touch()


def select_images():
    """
    Choisit N_IMAGES images du dossier val, le même nombre pour chaque classe.
    Le tirage utilise une graine fixe : on obtient toujours les mêmes images.
    Renvoie deux listes alignées : chemins (relatifs à la racine du projet) et labels (0 à 9).
    """
    rng = random.Random(config.SEED)
    per_class = config.N_IMAGES // len(config.CLASS_NAMES)
    paths, labels = [], []
    for label, wnid in enumerate(sorted(config.CLASS_NAMES)):
        class_dir = config.IMAGENETTE_DIR / "val" / wnid
        # Extension testée sans tenir compte de la casse (.JPEG, .jpeg, .jpg)
        class_files = sorted(p for p in class_dir.glob("*") if p.suffix.lower() in {".jpeg", ".jpg"})
        if len(class_files) < per_class:
            raise RuntimeError(f"{class_dir} : {len(class_files)} images trouvées, {per_class} demandées")
        chosen = sorted(rng.sample(class_files, per_class))
        paths += [p.relative_to(config.ROOT).as_posix() for p in chosen]
        labels += [label] * per_class
    return paths, labels


def get_resize_crop():
    """Petit côté redimensionné à 224, puis recadrage central 224x224."""
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(config.IMAGE_SIZE),
    ])


def get_transform():
    """Transformation donnée au modèle : redimensionnement/recadrage, tenseur, normalisation ImageNet."""
    return transforms.Compose([
        get_resize_crop(),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def load_display_image(path):
    """Image telle que vue par le modèle (même recadrage), sans normalisation : pour l'affichage."""
    image = Image.open(config.ROOT / path).convert("RGB")
    return np.asarray(get_resize_crop()(image))


def load_features(tag, kind):
    """Ouvre un tableau de features en cache en memmap : seules les lignes utilisées sont lues."""
    return np.load(config.FEATURES_DIR / f"{tag}_{kind}.npy", mmap_mode="r")


class ImageListDataset(Dataset):
    """Dataset minimal : une liste de chemins d'images et leurs labels."""

    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # .convert("RGB") : quelques images d'ImageNet sont en niveaux de gris
        image = Image.open(config.ROOT / self.paths[i]).convert("RGB")
        return self.transform(image), self.labels[i]


def get_loader(paths, labels):
    """DataLoader SANS mélange : l'ordre des features doit rester celui de `paths`."""
    dataset = ImageListDataset(paths, labels, get_transform())
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                      num_workers=config.NUM_WORKERS)
