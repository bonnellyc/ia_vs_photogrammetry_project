# IA vs Photogrammétrie - 3D Point Cloud Benchmark

## Objectif
Comparer des nuages de points générés par IA (Dust3r, Must3r...) à ceux générés par photogrammétrie, selon plusieurs métriques (C2C, Hausdorff, Chamfer) et valeurs de N images.


## 📁 Structure du projet
- `data/` : images, références, reconstructions
- `models/` : code source des modèles (repo GitHub)
- `scripts/` : scripts d’alignement, d’inférence, d’analyse
- `results/` : métriques et visualisations
- `config/` : configs par modèle ou expérience

## Exécution
```bash
bash scripts/model_wrappers/<model>_runner.py
```

## 📁 Structure des résultats

```
results/
│
├── metrics/                         # Données brutes métriques
│   ├── Dust3r/
│   │   ├── scene01.csv
│   │   ├── scene02.csv
│   ├── Must3r/
│   │   ├── scene01.csv
│   ├── Spann3r/
│       └── ...
│
├── plots/                           # Visualisations
│   ├── Dust3r_scene01_error_vs_N.png
│   ├── comparison_C2C_all_models.png
│
├── summary/                         # Tableaux d’ensemble
│   ├── metrics_summary.csv          # Fusion de tous les .csv
│   ├── metrics_summary.json         # (optionnel)
│   └── time_performance.csv         # Temps par N et modèle
```

Cette structure permet à la fois une analyse fine par modèle/scène, et une vue globale des performances sur l’ensemble du benchmark.

