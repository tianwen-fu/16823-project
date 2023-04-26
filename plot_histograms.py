import os
import pickle
from argparse import ArgumentParser

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from capture_frames import DepthData


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("data_file", type=str)
    return parser.parse_args()


def map_gray_image(gray_image, low, high, camera_matrix, distortion_coefficients):
    gray_image = gray_image.astype(np.float32).clip(low, high)
    gray_image = (gray_image - low) / (high - low) * 255
    gray_image = np.asarray(gray_image, dtype=np.uint8)
    gray_image = cv2.undistort(gray_image, camera_matrix, distortion_coefficients)
    return gray_image


def preprocess_frame(
    frame: DepthData, gray_mapping, camera_matrix, distortion_coefficients
):
    low, high = gray_mapping
    coords = cv2.undistort(frame.coords, camera_matrix, distortion_coefficients)
    coords = frame.coords * 1000  # meters to mm
    confidence = frame.confidence.reshape(-1)
    coords_valid = coords.reshape(-1, 3)
    coords_valid = coords_valid[confidence > 0, :]
    gray_image = map_gray_image(
        frame.grayscale, low, high, camera_matrix, distortion_coefficients
    )
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    point_colors = gray_image.reshape(-1, 3)[confidence > 0, :]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coords_valid))
    pcd.colors = o3d.utility.Vector3dVector(point_colors.astype(np.float32) / 255)
    return gray_image, coords, pcd


def estimate_extrinsics_by_procrustes(
    corners: np.ndarray,
    object_points: np.ndarray,
    point_cloud: np.ndarray,
):
    # corners: (N, 2)
    # point_cloud: (H, W, 3)
    image_points = corners.astype(np.int32).copy()
    image_points[:, 0] = np.clip(image_points[:, 0], 0, point_cloud.shape[1] - 1)
    image_points[:, 1] = np.clip(image_points[:, 1], 0, point_cloud.shape[0] - 1)
    # camera_points = point_cloud[image_points[:, 1], image_points[:, 0]]
    camera_points = point_cloud[image_points[:, 1], image_points[:, 0], :]
    assert camera_points.shape == object_points.shape
    camera_centroid = np.mean(camera_points, axis=0)  # (3,)
    world_centroid = np.mean(object_points, axis=0)  # (3,)
    camera_centered = camera_points - camera_centroid.reshape(1, 3)
    world_centered = object_points - world_centroid.reshape(1, 3)
    covariance_matrix = camera_centered.T @ world_centered
    U, _, VT = np.linalg.svd(covariance_matrix)
    rotation_matrix = VT.T @ U.T
    translation = world_centroid - camera_centroid @ rotation_matrix.T

    corner_points_world = camera_points @ rotation_matrix.T + translation.reshape(1, 3)
    errors = np.linalg.norm(corner_points_world - object_points, axis=1)  # (N,)

    return errors.mean(), rotation_matrix, translation


def get_object_points(dims, size):
    nx, ny = dims
    xx, yy = np.meshgrid(
        np.arange(nx - 1) * size, np.arange(ny - 1) * size
    )  # 15 mm blocks, 15 x 10
    zz = np.zeros_like(xx)
    object_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return object_points


def get_perblock_estimates(world_pts, confidence, checkerboard_dim, block_size):
    estimates = []
    nx, ny = checkerboard_dim
    xs = np.arange(nx) * block_size
    ys = np.arange(ny) * block_size
    for x_min, x_max in zip(xs, xs[1:]):
        row_estimates = []
        for y_min, y_max in zip(ys, ys[1:]):
            row_estimates.append(
                world_pts[
                    (confidence > 0)
                    & (world_pts[..., 0] > x_min)
                    & (world_pts[..., 0] < x_max)
                    & (world_pts[..., 1] > y_min)
                    & (world_pts[..., 1] < y_max)
                    & (world_pts[..., 2] > -10)  # some regularity conditions
                    & (world_pts[..., 2] < 10)
                ]
            )
            # print(x_min, x_max, y_min, y_max, row_estimates[-1].shape)
        estimates.append(row_estimates)
    return estimates


def collect_black_and_white_z(estimates, points_per_block):
    black_z = np.array([], dtype=np.float32)
    white_z = np.array([], dtype=np.float32)
    is_black_array = (
        []
    )  # return this array for visualization (and more importantly, bug detection)
    is_black = True  # first block is black
    for row in estimates:
        is_black_row = []
        for block in row:
            block_z = block[..., 2]
            block_z = np.random.choice(block_z, points_per_block)
            if is_black:
                black_z = np.concatenate([black_z, block_z])
            else:
                white_z = np.concatenate([white_z, block_z])
            is_black_row.append(is_black)
            is_black = not is_black
        if len(row) % 2 == 0:
            is_black = not is_black  # each row has different starting color
        is_black_array.append(is_black_row)
    return black_z, white_z, is_black_array


def plot_depth_histograms(perblock_estimates, dims, prefix):
    black_z, white_z, is_black = collect_black_and_white_z(perblock_estimates, 200)
    fig = plt.figure(figsize=(4, 3), dpi=600)
    plt.hist(black_z, bins=100, alpha=0.5, label="black", density=True)
    plt.hist(white_z, bins=100, alpha=0.5, label="white", density=True)
    plt.legend()
    plt.savefig(f"{prefix}_overall.pdf")
    plt.close(fig)
    # also plot the shifted z
    fig = plt.figure(figsize=(4, 3), dpi=600)
    plt.hist(black_z - np.mean(white_z), bins=100, alpha=0.5, density=True)
    plt.xlim(-10, 10)
    plt.savefig(f"{prefix}_shifted_z.pdf")
    plt.close(fig)
    nx, ny = dims
    fig, axs = plt.subplots(ny - 2, nx - 2, figsize=(nx - 1, ny - 1), dpi=600)
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            if is_black[j][i]:
                ax.set_facecolor("black")
            else:
                ax.set_facecolor("white")
            ax.hist(perblock_estimates[j][i][..., 2], bins=100)
            ax.set_xlim(-10, 10)
    plt.savefig(f"{prefix}_depth_histograms.pdf")
    plt.close(fig)
    return black_z, white_z, is_black


def process_frame(frame: DepthData, prefix, metadata):
    dims = (15, 10)
    size = 15
    gray_mapping = metadata["gray_mapping"]
    camera_matrix = metadata["camera_matrix"]
    distortion_coefficients = metadata["distortion_coefficients"]
    gray_image, coords, pcd = preprocess_frame(
        frame, gray_mapping, camera_matrix, distortion_coefficients
    )
    cv2.imwrite(f"{prefix}_gray.png", gray_image)
    o3d.io.write_point_cloud(f"{prefix}.ply", pcd)
    object_points = get_object_points(dims, size)
    found, corners = cv2.findChessboardCorners(gray_image, (14, 9))
    if not found:
        plt.imshow(gray_image)
        plt.show()
        exit(1)
    errors, rotation, translation = estimate_extrinsics_by_procrustes(
        corners.reshape(-1, 2), object_points.reshape(-1, 3), coords
    )
    print(f"Frame {prefix} errors: {errors}")
    points_world = coords @ rotation.T + translation.reshape(1, 3)
    world_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(points_world.reshape(-1, 3))
    )
    world_pcd.colors = pcd.colors
    o3d.io.write_point_cloud(f"{prefix}_world.ply", world_pcd)
    perblock_estimates = get_perblock_estimates(
        points_world, frame.confidence, dims, size
    )
    black_z, white_z, is_black = plot_depth_histograms(perblock_estimates, dims, prefix)
    return errors, perblock_estimates, black_z, white_z, is_black


def main():
    args = parse_args()
    work_dir = os.path.dirname(__file__)
    work_dir = os.path.join(
        work_dir, "results", os.path.basename(args.data_file).strip(".pkl")
    )
    print("saving to", work_dir, "...")
    os.makedirs(work_dir, exist_ok=True)
    with open(args.data_file, "rb") as f:
        data = pickle.load(f)
    frame_stats = []
    for idx, frame in enumerate(data["frames"]):
        frame_stats.append(
            process_frame(frame, os.path.join(work_dir, f"{idx:03}"), data)
        )
    errors, perblock_estimates, black_z, white_z, is_black = zip(*frame_stats)
    with open(os.path.join(work_dir, "stats.pkl"), "wb") as f:
        pickle.dump(
            {
                "errors": errors,
                "perblock_estimates": perblock_estimates,
                "black_z": black_z,
                "white_z": white_z,
                "is_black": is_black,
            },
            f,
        )


if __name__ == "__main__":
    main()
