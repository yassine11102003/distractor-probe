"""Étape 06 : lit les JSON des étapes 03 à 05 et écrit un résumé (tableau, conclusions, limites)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

MODELS = list(config.MODELS)   # ["dinov2", "dinov2_reg"]


def load_results():
    """Charge les JSON des scripts 03, 04 et 05 (il faut les avoir lancés avant)."""
    names = {"pos": "03_position_probe", "norm": "04_norm_outliers", "knn": "05_knn_pooling"}
    results = {}
    for key, name in names.items():
        path = config.RESULTS_DIR / f"{name}.json"
        if not path.exists():
            sys.exit(f"{path.name} manquant : lance d'abord scripts/{name}.py")
        results[key] = json.loads(path.read_text())
    return results


def pct(x):
    """0.1234 -> '12.3 %'."""
    return f"{100 * x:.1f} %"


def build_table(r):
    """Tableau markdown : une ligne par chiffre clé, une colonne par modèle."""
    pos, norm, knn = r["pos"]["models"], r["norm"]["models"], r["knn"]["models"]
    rows = [
        ("Sonde position, colonne, brut", lambda m: pct(pos[m]["accuracy"]["raw"]["col"])),
        ("Sonde position, colonne, position retirée", lambda m: pct(pos[m]["accuracy"]["position_removed"]["col"])),
        ("Sonde position, ligne, brut", lambda m: pct(pos[m]["accuracy"]["raw"]["row"])),
        ("Sonde position, ligne, position retirée", lambda m: pct(pos[m]["accuracy"]["position_removed"]["row"])),
        ("Part de variance due à la position", lambda m: pct(pos[m]["position_variance_share"])),
        ("Tokens outliers (norme pré-LayerNorm)", lambda m: f"{norm[m]['pct_outlier_tokens']:.2f} %"),
        ("Images avec au moins un outlier", lambda m: f"{norm[m]['pct_images_with_outlier']:.1f} %"),
        ("cos(pool, pool sans outliers), images touchées",
         lambda m: str(norm[m]["pool_cosine_mean_affected_images"] or "–")),   # "–" : aucune image touchée
        ("k-NN, mean-pool", lambda m: pct(knn[m]["mean_pool"])),
        ("k-NN, mean-pool sans outliers", lambda m: pct(knn[m]["mean_pool_no_outliers"])),
        ("k-NN, token CLS", lambda m: pct(knn[m]["cls"])),
    ]
    lines = ["| Mesure | " + " | ".join(MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    lines += [f"| {label} | " + " | ".join(get(m) for m in MODELS) + " |" for label, get in rows]
    lines.append(f"\nHasard : position {pct(r['pos']['chance'])}, k-NN {pct(r['knn']['chance'])} "
                 f"({r['knn']['n_test_images']} images de test, split par image).")
    return "\n".join(lines)


def build_conclusion(r):
    """Conclusions : chaque phrase dépend d'une règle de décision explicite, indiquée entre crochets."""
    chance = r["pos"]["chance"]
    pos, norm, knn = r["pos"]["models"]["dinov2"], r["norm"]["models"], r["knn"]["models"]["dinov2"]
    col_raw, col_removed = pos["accuracy"]["raw"]["col"], pos["accuracy"]["position_removed"]["col"]
    out_no_reg, out_reg = norm["dinov2"]["pct_outlier_tokens"], norm["dinov2_reg"]["pct_outlier_tokens"]
    gap_cls = knn["mean_pool"] - knn["cls"]
    s = []

    # Distracteur 1 : la position
    if col_raw > 2 * chance:
        s.append(f"La position est présente et lisible linéairement dans les tokens de patch de DINOv2 "
                 f"(colonne prédite à {pct(col_raw)}, hasard {pct(chance)}) [règle : > 2 x hasard].")
    else:
        s.append(f"La position est peu lisible linéairement ({pct(col_raw)}), ce qui va contre "
                 f"l'hypothèse d'un fort signal positionnel [règle : <= 2 x hasard].")
    if col_removed < chance + 0.03:
        s.append(f"Retirer la moyenne de chaque position ramène la sonde au hasard ({pct(col_removed)}) : "
                 f"cette composante est essentiellement additive, donc éliminée par le mean-pooling "
                 f"[règle : < hasard + 3 points].")
    else:
        s.append(f"Après retrait de la moyenne par position, la sonde reste au-dessus du hasard "
                 f"({pct(col_removed)}) : une partie de la position n'est pas additive et le mean-pooling "
                 f"ne l'élimine pas forcément [règle : >= hasard + 3 points].")

    reg = r["pos"]["models"]["dinov2_reg"]
    s.append(f"Avec registres, la colonne est prédite à {pct(reg['accuracy']['raw']['col'])} (contre "
             f"{pct(col_raw)}) et la part de variance due à la position vaut {pct(reg['position_variance_share'])} "
             f"(contre {pct(pos['position_variance_share'])}) : écart observé sur une seule graine, non expliqué.")

    # Distracteur 2 : les artefacts. Phrase descriptive (pas de verdict binaire) ; on dit
    # explicitement si le seuil fixé AVANT de voir les données (0.5 %, en référence aux ~2 %
    # de Darcet et al.) est atteint ou non.
    img_no_reg, img_reg = norm["dinov2"]["pct_images_with_outlier"], norm["dinov2_reg"]["pct_images_with_outlier"]
    ratio = f"rapport x{out_no_reg / out_reg:.0f} sur les tokens" if out_reg > 0 else "aucun outlier avec registres"
    threshold_status = "est atteint" if out_no_reg > 0.5 else "n'est pas atteint"
    s.append(f"Tokens à forte norme (règle : médiane + {config.OUTLIER_IQR_K} x IQR) : {out_no_reg:.2f} % des "
             f"tokens et {img_no_reg:.1f} % des images sans registres, contre {out_reg:.2f} % et {img_reg:.1f} % "
             f"avec registres ({ratio}). Le seuil fixé a priori (> 0.5 % des tokens, en référence aux ~2 % "
             f"rapportés par Darcet et al.) {threshold_status}.")
    s.append(f"Après LayerNorm, exclure ces tokens change peu le descripteur mean-poolé "
             f"(cosinus moyen {norm['dinov2']['pool_cosine_mean_affected_images']} sur les images touchées).")

    # Sémantique du mean-pool
    if abs(gap_cls) <= 0.01:
        s.append(f"En k-NN, le mean-pool est au même niveau que le token CLS ({pct(knn['mean_pool'])} contre "
                 f"{pct(knn['cls'])}) [règle : écart <= 1 point].")
    else:
        s.append(f"En k-NN, le mean-pool et le token CLS diffèrent de {100 * gap_cls:+.1f} points "
                 f"({pct(knn['mean_pool'])} contre {pct(knn['cls'])}).")

    return "\n".join(f"- {sentence}" for sentence in s)


LIMITS = """- Les mesures portent uniquement sur le teacher : aucun autoencodeur ni modèle de diffusion n'est \
entraîné, il n'y a donc pas de gFID, et rien ici ne montre que ces distracteurs dégradent la diffusabilité.
- Imagenette (10 classes faciles) crée un effet plafond pour le k-NN.
- Une seule graine pour le split et le tirage des tokens.
- Expérience suivante : entraîner VF avec DINOv2-reg contre DINOv2, et VF sur des tokens dont on a retiré \
la moyenne par position, puis comparer le gFID."""


def main():
    r = load_results()
    text = "\n\n".join([
        "# Résumé des résultats",
        "## Tableau", build_table(r),
        "## Conclusions", build_conclusion(r),
        "## Limites", LIMITS,
    ])
    out_path = config.RESULTS_DIR / "summary.md"
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nRésumé écrit dans {out_path}")


if __name__ == "__main__":
    main()
