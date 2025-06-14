import open3d as o3d
import numpy as np
import argparse
import os
import csv
from scipy.spatial import KDTree
from scipy.spatial.distance import directed_hausdorff

def compute_c2c_metrics(source_pts, target_pts):
    tree_target = KDTree(target_pts)
    dist_src_to_tgt, _ = tree_target.query(source_pts)

    tree_source = KDTree(source_pts)
    dist_tgt_to_src, _ = tree_source.query(target_pts)

    chamfer = (np.mean(dist_src_to_tgt) + np.mean(dist_tgt_to_src)) / 2
    c2c_mean = np.mean(dist_src_to_tgt)
    c2c_median = np.median(dist_src_to_tgt)
    rms = np.sqrt(np.mean(dist_src_to_tgt ** 2))
    hausdorff = max(
        directed_hausdorff(source_pts, target_pts)[0],
        directed_hausdorff(target_pts, source_pts)[0]
    )

    return {
        "C2C_mean": c2c_mean,
        "C2C_median": c2c_median,
        "RMS": rms,
        "Chamfer": chamfer,
        "Hausdorff": hausdorff
    }

def main(args):
    print("Chargement des nuages...")
    aligned = o3d.io.read_point_cloud(args.aligned)
    reference = o3d.io.read_point_cloud(args.reference)

    aligned_pts = np.asarray(aligned.points)
    reference_pts = np.asarray(reference.points)

    print("Calcul des métriques...")
    metrics = compute_c2c_metrics(aligned_pts, reference_pts)
    metrics["N_images"] = args.N
    metrics["Inference_time"] = args.inference_time

    header = ["N_images", "C2C_mean", "C2C_median", "RMS", "Chamfer", "Hausdorff", "Inference_time"]
    row = [metrics[h] for h in header]

    out_dir = os.path.join("results", args.model, "metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out_csv or os.path.join(out_dir, f"{args.scene}.csv")

    file_exists = os.path.isfile(out_path)
    with open(out_path, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

    print(f"Métriques ajoutées à : {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcul des métriques entre nuages aligné et référence")
    parser.add_argument("--aligned", required=True, help="Fichier du nuage aligné (PLY)")
    parser.add_argument("--reference", required=True, help="Fichier du nuage de référence (PLY)")
    parser.add_argument("--model", required=True, help="Nom du modèle IA utilisé")
    parser.add_argument("--scene", required=True, help="Nom de la scène")
    parser.add_argument("--N", type=int, required=True, help="Nombre d'images utilisées pour inférence")
    parser.add_argument("--inference_time", type=float, default=-1.0, help="Temps d'inférence (secondes)")
    parser.add_argument("--out_csv", type=str, help="Fichier de sortie CSV (optionnel)")
    args = parser.parse_args()

    main(args)
