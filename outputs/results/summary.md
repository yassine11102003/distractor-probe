# Résumé des résultats

## Tableau

| Mesure | dinov2 | dinov2_reg |
|---|---|---|
| Sonde position, colonne, brut | 54.0 % | 41.0 % |
| Sonde position, colonne, position retirée | 7.0 % | 6.8 % |
| Sonde position, ligne, brut | 50.6 % | 35.4 % |
| Sonde position, ligne, position retirée | 8.0 % | 7.7 % |
| Part de variance due à la position | 6.8 % | 5.6 % |
| Tokens outliers (norme pré-LayerNorm) | 0.40 % | 0.04 % |
| Images avec au moins un outlier | 79.2 % | 1.5 % |
| cos(pool, pool sans outliers), images touchées | 0.99993 | 0.99882 |
| k-NN, mean-pool | 98.2 % | 98.3 % |
| k-NN, mean-pool sans outliers | 98.2 % | 98.3 % |
| k-NN, token CLS | 99.5 % | 99.3 % |

Hasard : position 6.2 %, k-NN 10.0 % (1000 images de test, split par image).

## Conclusions

- La position est présente et lisible linéairement dans les tokens de patch de DINOv2 (colonne prédite à 54.0 %, hasard 6.2 %) [règle : > 2 x hasard].
- Retirer la moyenne de chaque position ramène la sonde au hasard (7.0 %) : cette composante est essentiellement additive, donc éliminée par le mean-pooling [règle : < hasard + 3 points].
- Avec registres, la colonne est prédite à 41.0 % (contre 54.0 %) et la part de variance due à la position vaut 5.6 % (contre 6.8 %) : écart observé sur une seule graine, non expliqué.
- Tokens à forte norme (règle : médiane + 3.0 x IQR) : 0.40 % des tokens et 79.2 % des images sans registres, contre 0.04 % et 1.5 % avec registres (rapport x11 sur les tokens). Le seuil fixé a priori (> 0.5 % des tokens, en référence aux ~2 % rapportés par Darcet et al.) n'est pas atteint.
- Après LayerNorm, exclure ces tokens change peu le descripteur mean-poolé (cosinus moyen 0.99993 sur les images touchées).
- En k-NN, le mean-pool et le token CLS diffèrent de -1.3 points (98.2 % contre 99.5 %).

## Limites

- Les mesures portent uniquement sur le teacher : aucun autoencodeur ni modèle de diffusion n'est entraîné, il n'y a donc pas de gFID, et rien ici ne montre que ces distracteurs dégradent la diffusabilité.
- Imagenette (10 classes faciles) crée un effet plafond pour le k-NN.
- Une seule graine pour le split et le tirage des tokens.
- Expérience suivante : entraîner VF avec DINOv2-reg contre DINOv2, et VF sur des tokens dont on a retiré la moyenne par position, puis comparer le gFID.