"""Configuration du projet : tous les paramètres au même endroit."""
from pathlib import Path

# --- Chemins ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
FEATURES_DIR = OUTPUTS_DIR / "features"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RESULTS_DIR = OUTPUTS_DIR / "results"

# --- Données ---
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
IMAGENETTE_DIR = DATA_DIR / "imagenette2-320"
N_IMAGES = 2000                         # 200 images par classe
IMAGE_SIZE = 224
PATCH_SIZE = 14
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE    # 16 -> 256 tokens de patch
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Classes d'Imagenette : dossier WordNet -> nom lisible ---
CLASS_NAMES = {
    "n01440764": "tench",           "n02102040": "English springer",
    "n02979186": "cassette player", "n03000684": "chain saw",
    "n03028079": "church",          "n03394916": "French horn",
    "n03417042": "garbage truck",   "n03425413": "gas pump",
    "n03445777": "golf ball",       "n03888257": "parachute",
}

# --- Modèles ---
HUB_REPO = "facebookresearch/dinov2"
MODELS = {
    "dinov2": "dinov2_vitl14",          # sans registres
    "dinov2_reg": "dinov2_vitl14_reg",  # avec registres
}
EMBED_DIM = 1024

# --- Calcul ---
SEED = 0
BATCH_SIZE = 64
NUM_WORKERS = 2

# --- Visualisations (scripts 02 et 04) ---
VIS_IMAGE_INDICES = [0, 500, 1000, 1500]   # 4 images de 4 classes différentes

# --- Analyses (scripts 03 à 05) ---
TEST_FRACTION = 0.5
PROBE_TOKENS_PER_IMAGE = 16   # tokens tirés au hasard par image pour la sonde de position
PROBE_MAX_ITER = 300
OUTLIER_IQR_K = 3.0           # outlier si norme > médiane + k * IQR (clôture "extrême" de Tukey)
KNN_K = 20
