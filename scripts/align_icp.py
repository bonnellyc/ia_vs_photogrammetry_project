# Script d'alignement ICP avec mise à l'échelle
import open3d as o3d
import numpy as np
import argparse
import os

def preprocess_point_cloud(pcd: open3d.geometry.PointCloud, voxel_size: float):
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 2, max_nn=30))
    return pcd

def main(args):
    print("Chargement des nuages de points...")
    source = o3d.io.read_point_cloud(args.source)
    target = o3d.io.read_point_cloud(args.target)

    source = preprocess_point_cloud(source, args.voxel_size)
    target = preprocess_point_cloud(target, args.voxel_size)

    print("Alignement initial par correspondances globales (avec scale)...")
    result = o3d.pipelines.registration.registration_generalized_icp(
        source, target,
        max_correspondence_distance=args.voxel_size * 2,
        estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )

    print("Transformation estimée (incluant échelle) :\n", result.transformation)
    print("Fitness : {:.4f}, Inlier RMSE : {:.6f}".format(result.fitness, result.inlier_rmse))

    if args.output:
        aligned = source.transform(result.transformation)
        o3d.io.write_point_cloud(args.output, source)
        print(f"Nuage aligné sauvegardé dans : {args.output}")

    if args.save_transform:
        np.save(args.save_transform, result.transformation)
        print(f"Matrice de transformation sauvegardée dans : {args.save_transform}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alignement ICP avec mise à l'échelle")
    parser.add_argument("--source", required=True, help="Fichier PLY du nuage IA (source)")
    parser.add_argument("--target", required=True, help="Fichier PLY du nuage photogrammétrique (référence)")
    parser.add_argument("--voxel_size", type=float, default=0.005, help="Taille de voxel pour downsampling")
    parser.add_argument("--output", type=str, help="Chemin du nuage source aligné (sortie)")
    parser.add_argument("--save_transform", type=str, help="Chemin pour sauvegarder la matrice de transformation (.npy)")
    args = parser.parse_args()

    main(args)
