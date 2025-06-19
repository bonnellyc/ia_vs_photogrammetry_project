import argparse
import open3d as o3d
import numpy as np
from pathlib import Path

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

def infer(image_path: str):
    """
        image_path: chemin vers les images

        return : point cloud généré par le model
    """
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    # load the model
    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)

    # load images
    images = load_images(image_path, size=512)

    # prepare input for Dust3r -> pairs of images
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)

    pts3d = scene.get_pts3d() # pointmap complete
    confidence_masks = scene.get_masks() # score de confiance pour chaque points


    # save the pointmaps
    pts3d_array = np.array([p.detach().cpu().numpy() for p in pts3d])
    filtered_pts3d = [p[m.cpu().numpy()] for p, m in zip(pts3d_array, confidence_masks)]

    # Concaténation des points 3D
    all_pts = np.vstack(filtered_pts3d)

    # Creation du nuage de point
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(all_pts)

    return point_cloud

def run_dust3r(input_dir, output_path):
    """
        input_dir: chemin contenant toute les images
        output_dir: chemin ou enregistrer le point_cloud au format.ply
    """
    result_pcd = infer(input_dir)
    o3d.io.write_point_cloud(output_path, result_pcd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--N", type=int, required=False)
    args = parser.parse_args()

    run_dust3r(args.input_dir, args.output)