from subprocess import check_output
import uuid
import numpy as np
import pycolmap
from pathlib import Path
import cv2


def read_sparse_reconstruction(rec_path: Path) -> None:
    print("Reading sparse COLMAP reconstruction")
    reconstruction = pycolmap.Reconstruction(rec_path)
    cameras = reconstruction.cameras
    images = reconstruction.images
    points3D = reconstruction.points3D

    poses = []
    projs = []

    for image in sorted(images.values(), key=lambda im: im.name):
        camera = cameras[image.camera_id]

        pose = np.eye(4)
        pose[:3, 3] = image.cam_from_world.translation
        pose[:3, :3] = image.cam_from_world.rotation.matrix()
        pose = np.linalg.inv(pose)

        cam_params = camera.params

        proj = np.eye(3)

        if len(cam_params) == 4:
            proj[0, 0] = cam_params[0]
            proj[1, 1] = cam_params[1]
            proj[0, 2] = cam_params[2]
            proj[1, 2] = cam_params[3]
        elif len(cam_params) == 3:
            proj[0, 0] = cam_params[0]
            proj[1, 1] = cam_params[0]
            proj[0, 2] = cam_params[1]
            proj[1, 2] = cam_params[2]

        poses.append(pose)
        projs.append(proj)

    return poses, projs


def get_poses_from_colmap(
    imgs,
    colmap_command_template,
    scene_name,
    out_dir=None,
    tmp_dir="/tmp/colmap-io"
):
    tmp_dir = tmp_dir + str(uuid.uuid4())

    # Create a temporary directory
    tmp_dir = Path(tmp_dir) / scene_name
    tmp_dir.mkdir(exist_ok=True, parents=True)

    out_dir = Path(out_dir) / scene_name
    out_dir.mkdir(exist_ok=True, parents=True)

    img_dir = tmp_dir / "images"
    img_dir.mkdir(exist_ok=True)

    print("Saving images to tmp dir")

    for i, img in enumerate(imgs):
        cv2.imwrite(str(img_dir / f"{i:04d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Run COLMAP
    print("Running COLMAP command")

    colmap_command = colmap_command_template.format(str(tmp_dir))

    print(colmap_command)

    try:
        colmap_cli_output = check_output(colmap_command, shell=True)

        if out_dir is not None:
            # Copy results to out dir
            colmap_output = Path(tmp_dir) / "sparse"

            copy_command = f"cp -r {colmap_output} {out_dir}"

            print("Copying results to out dir")
            print(copy_command)

            check_output(copy_command, shell=True)
        else:
            out_dir = tmp_dir

        if (out_dir / "sparse" / "0").exists():
            out_dir = out_dir / "sparse" / "0"
        else:
            out_dir = out_dir / "sparse"

        # Load poses
        poses, projs = read_sparse_reconstruction(out_dir)

    except:
        poses, projs = None, None
        colmap_cli_output = None

    print("Clearing directory")

    # Clear tmp dir
    check_output(f"rm -r {tmp_dir}", shell=True)

    return poses, projs, colmap_cli_output
