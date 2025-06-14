import argparse
import os
from pathlib import Path

def run_dust3r(input_dir, output_path):
    # Exécuter le modèle Dust3r ici
    # Exemple fictif : suppose que dust3r_infer est une fonction de leur repo
    from dust3r.inference import infer
    images = sorted(Path(input_dir).glob("*.jpg"))  # ou *.png
    result_pcd = infer(images)  # résultat Open3D ou numpy
    result_pcd.save(output_path)  # ou o3d.io.write_point_cloud(output_path, result_pcd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--N", type=int, required=False)
    args = parser.parse_args()

    run_dust3r(args.input_dir, args.output)