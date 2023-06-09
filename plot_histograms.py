import os
import pickle
from argparse import ArgumentParser
from contextlib import contextmanager
from copy import deepcopy
from functools import partial

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.neighbors import KernelDensity

from capture_frames import DepthData


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("data_file", type=str)
    return parser.parse_args()


@contextmanager
def plot_and_save(
    name,
    work_dir=None,
    prefix=None,
    figsize=(4, 2.5),
    dpi=600,
    show=False,
    **fig_kwargs,
):
    if work_dir is not None and prefix is not None:
        raise ValueError("Cannot specify both work_dir and prefix")
    elif work_dir is not None:
        _path = lambda name: os.path.join(work_dir, name)
    elif prefix is not None:
        _path = lambda name: f"{prefix}_{name}"
    else:
        raise ValueError("Must specify either work_dir or prefix")

    fig = plt.figure(figsize=figsize, dpi=dpi, **fig_kwargs)
    data = {}

    def _record_data(k, v):
        data[k] = v

    yield _record_data
    plt.savefig(
        _path(f"{name}.pdf"),
        bbox_inches="tight",
        dpi=dpi,
        pad_inches=0,
    )
    # also save a png image, in case we want to use it in a presentation
    plt.savefig(
        _path(f"{name}.png"),
        bbox_inches="tight",
        dpi=dpi,
        pad_inches=0,
    )
    if data:
        with open(_path(f"{name}.pkl"), "wb") as f:
            pickle.dump(data, f)
    if show:
        plt.show()
    plt.close(fig)


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
    point_colors = point_colors.astype(np.float32) / 255
    pcd.colors = o3d.utility.Vector3dVector(point_colors.copy())
    return gray_image, coords, pcd, point_colors


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
    assert (
        camera_points.shape == object_points.shape
    ), f"{camera_points.shape} != {object_points.shape}"
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


def get_perblock_estimates(
    world_pts, color_image, confidence, checkerboard_dim, block_size
):
    assert (
        world_pts.shape[:2] == color_image.shape[:2]
    ), f"{world_pts.shape} {color_image.shape}"
    estimates = []
    nx, ny = checkerboard_dim
    xs = np.arange(nx) * block_size
    ys = np.arange(ny) * block_size
    for x_min, x_max in zip(xs, xs[1:]):
        row_estimates = []
        for y_min, y_max in zip(ys, ys[1:]):
            block_mask = (
                (confidence > 0)
                & (world_pts[..., 0] > x_min)
                & (world_pts[..., 0] < x_max)
                & (world_pts[..., 1] > y_min)
                & (world_pts[..., 1] < y_max)
                & (world_pts[..., 2] > -10)  # some regularity conditions
                & (world_pts[..., 2] < 10)
            )
            row_estimates.append((world_pts[block_mask], color_image[block_mask]))
            # print(x_min, x_max, y_min, y_max, row_estimates[-1].shape)
        estimates.append(row_estimates)
    return estimates


def collect_black_and_white_z(estimates, points_per_block):
    black_z = np.array([], dtype=np.float32)
    white_z = np.array([], dtype=np.float32)
    COLOR_THRESHOLD = 128
    ACCEPTING_RATIO = 0.6
    skipped_blocks = 0
    is_black_array = (
        []
    )  # return this array for visualization (and more importantly, bug detection)
    is_black = True  # first block is black
    for row in estimates:
        is_black_row = []
        for block in row:
            block_coords, block_colors = block

            # verify color masks
            block_black_mask = block_colors[:, 0] < COLOR_THRESHOLD  # (N,)

            if is_black:
                block_z = block_coords[block_black_mask, 2]
            else:
                block_z = block_coords[~block_black_mask, 2]

            if block_z.shape[0] > ACCEPTING_RATIO * block_coords.shape[0]:
                block_z = np.random.choice(block_z, points_per_block)
                if is_black:
                    black_z = np.concatenate([black_z, block_z])
                else:
                    white_z = np.concatenate([white_z, block_z])
            else:
                skipped_blocks += 1
            is_black_row.append(is_black)
            is_black = not is_black
        if len(row) % 2 == 0:
            is_black = not is_black  # each row has different starting color
        is_black_array.append(is_black_row)
    return black_z, white_z, is_black_array, skipped_blocks


def estimate_posterior_eps(shifted_z, z_argmax):
    # by document, pico flexx uses 850nm wavelength
    freq = 1 / (850e-9)
    phis = 4 * np.pi * shifted_z * freq / 3e8
    estimated_albedo_ratio = 4 * np.pi * z_argmax * freq / 3e8
    sine_values = 1 / estimated_albedo_ratio * np.sin(phis)
    valid_mask = (-1 < sine_values) & (sine_values < 1)
    eps_minus_phis = np.arcsin(sine_values[valid_mask])  # this is eps minus phi_hat
    eps_minus_phis -= phis[valid_mask]
    return eps_minus_phis


def plot_depth_histograms(perblock_estimates, dims, prefix):
    _plot_and_save = partial(plot_and_save, prefix=prefix)
    black_z, white_z, is_black, skipped_blocks = collect_black_and_white_z(
        perblock_estimates, 200
    )
    print(f"{prefix}: Skipped {skipped_blocks} blocks due to not enough points")
    with _plot_and_save("overall"):
        plt.hist(black_z, bins=100, alpha=0.5, label="black", density=True)
        plt.hist(white_z, bins=100, alpha=0.5, label="white", density=True)
        plt.legend()
    # also plot the shifted z
    with _plot_and_save("shifted_z"):
        shifted_z = black_z - np.mean(white_z)
        plt.hist(shifted_z, bins=100, alpha=0.5, density=True)
        plt.xlabel(r"$\hat{z} - z$")
        plt.ylabel("Density")
        plt.xlim(-10, 10)

    # estimate albedo ratio
    z_linspace = np.linspace(
        np.percentile(shifted_z, 5), np.percentile(shifted_z, 95), 1000
    )
    kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(shifted_z.reshape(-1, 1))
    log_prob = kde.score_samples(z_linspace.reshape(-1, 1))
    z_argmax = z_linspace[np.argmax(log_prob)]
    eps_minus_phis = estimate_posterior_eps(shifted_z, z_argmax)
    with _plot_and_save("posterior_eps_hist") as record_data:
        plt.hist(eps_minus_phis, bins=100, density=True)
        plt.xlabel(r"$\epsilon - \varphi$")
        plt.ylabel("Density")
        record_data("eps_minus_phis", eps_minus_phis)

    nx, ny = dims
    fig, axs = plt.subplots(ny - 2, nx - 2, figsize=(nx - 1, ny - 1), dpi=600)
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            if is_black[j][i]:
                ax.set_facecolor("black")
            else:
                ax.set_facecolor("white")
            estimate, image = perblock_estimates[j][i]
            ax.hist(estimate[..., 2], bins=100)
            ax.set_xlim(-10, 10)
    plt.savefig(f"{prefix}_depth_histograms.pdf", bbox_inches="tight", pad_inches=0)
    plt.savefig(f"{prefix}_depth_histograms.png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return black_z, white_z, is_black, eps_minus_phis


def process_frame(frame: DepthData, gray_mapping, prefix, metadata):
    dims = metadata["dims"]
    size = metadata["size"]
    camera_matrix = metadata["camera_matrix"]
    distortion_coefficients = metadata["distortion_coefficients"]
    gray_image, coords, pcd, point_colors = preprocess_frame(
        frame, gray_mapping, camera_matrix, distortion_coefficients
    )
    cv2.imwrite(f"{prefix}_gray.png", gray_image)
    o3d.io.write_point_cloud(f"{prefix}.ply", pcd)
    object_points = get_object_points(dims, size)
    found, corners = cv2.findChessboardCorners(gray_image, (dims[0] - 1, dims[1] - 1))
    if not found:
        print("Could not find pattern, ", prefix)
        exit(1)
    errors, rotation, translation = estimate_extrinsics_by_procrustes(
        corners.reshape(-1, 2), object_points.reshape(-1, 3), coords
    )
    print(f"Frame {prefix} errors: {errors}")
    points_world = coords @ rotation.T + translation.reshape(1, 3)
    world_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(points_world[frame.confidence > 0].reshape(-1, 3))
    )
    world_pcd.colors = o3d.utility.Vector3dVector(point_colors)
    o3d.io.write_point_cloud(f"{prefix}_world.ply", world_pcd)
    perblock_estimates = get_perblock_estimates(
        points_world, gray_image, frame.confidence, dims, size
    )
    black_z, white_z, is_black, eps_minus_phis = plot_depth_histograms(
        perblock_estimates, dims, prefix
    )
    return (
        errors,
        perblock_estimates,
        black_z,
        white_z,
        is_black,
        eps_minus_phis,
        translation.flatten()[2],
    )


def generate_frame_stats_distribution(frame_stats, work_dir):
    _plot_and_save = partial(plot_and_save, work_dir=work_dir)
    xs = np.linspace(-10, 10, 1000)
    with _plot_and_save("black_z_distribution") as record_data:
        # compare the probability curves
        for idx, frame in enumerate(frame_stats):
            if frame["errors"] > 10:
                continue
            shifted_z = frame["black_z"] - frame["white_z"].mean()
            kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(
                shifted_z.reshape(-1, 1)
            )
            log_prob = kde.score_samples(xs.reshape(-1, 1))
            probs = np.exp(log_prob)
            stats_to_record = dict(
                probs=probs,
                raw=shifted_z,
                mean=shifted_z.mean(),
                std=shifted_z.std(),
            )
            if "different_depth" in work_dir:
                label = f'{frame["depth"]:.0f}'
                record_data(frame["depth"], stats_to_record)
            else:
                label = f"{idx // 2}_{idx % 2}"
                record_data(idx, stats_to_record)
            plt.plot(xs, probs, label=label)
        plt.legend()
    # also plot an overall plot of shifted z
    with _plot_and_save("overall_shifted_z") as record_data:
        # collect shifted z's
        shifted_z = np.array([], dtype=frame_stats[0]["black_z"].dtype)
        for frame in frame_stats:
            this_shifted_z = frame["black_z"] - frame["white_z"].mean()
            shifted_z = np.concatenate([shifted_z, this_shifted_z])
        plt.hist(
            shifted_z,
            bins=100,
            weights=np.ones_like(shifted_z) / len(shifted_z),
        )
        record_data("shifted_z", shifted_z)
        plt.ylabel("Frequency")
    with _plot_and_save("overall_shifted_z_kde") as record_data:
        kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(
            shifted_z.reshape(-1, 1)
        )
        log_prob = kde.score_samples(xs.reshape(-1, 1))
        probs = np.exp(log_prob)
        record_data("shifted_z_probs", probs)
        plt.plot(xs, probs)

    # posterior
    phis = np.linspace(-2, 2, 1000)
    with _plot_and_save("eps_distribution") as record_data:
        for idx, frame in enumerate(frame_stats):
            if frame["errors"] > 10:
                continue
            kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(
                frame["eps_minus_phis"].reshape(-1, 1)
            )
            log_prob = kde.score_samples(phis.reshape(-1, 1))
            probs = np.exp(log_prob)
            stats_to_record = dict(probs=probs, raw=frame["eps_minus_phis"])
            if "different_depth" in work_dir:
                label = f'{frame["depth"]:.0f}'
                record_data(frame["depth"], stats_to_record)
            else:
                label = f"{idx // 2}_{idx % 2}"
                record_data(idx, stats_to_record)
            plt.plot(phis, probs, label=label)
    with _plot_and_save("overall_eps_distribution_kde") as record_data:
        eps_minus_phis = np.array([], dtype=frame_stats[0]["eps_minus_phis"].dtype)
        for frame in frame_stats:
            this_eps_minus_phis = frame["eps_minus_phis"]
            eps_minus_phis = np.concatenate([eps_minus_phis, this_eps_minus_phis])
        kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(
            eps_minus_phis.reshape(-1, 1)
        )
        log_prob = kde.score_samples(phis.reshape(-1, 1))
        probs = np.exp(log_prob)
        record_data("eps_minus_phis_probs", probs)
        plt.plot(phis, probs)


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

    # compatible with older version pkls
    gray_mapping_meta = data["gray_mapping"]
    if not isinstance(gray_mapping_meta[0], tuple):
        # extend it to the shape
        gray_mapping_meta = [gray_mapping_meta for _ in data["frames"]]

    # set dim and size
    if "7x5" in args.data_file:
        data["dims"] = (7, 5)
        data["size"] = 35.5
    elif "15x10" in args.data_file:
        data["dims"] = (15, 10)
        data["size"] = 15
    else:
        raise ValueError('Unknown pattern size, please specify "5x7" or "15x10"')

    for idx, (frame, gray_mapping) in enumerate(zip(data["frames"], gray_mapping_meta)):
        (
            errors,
            perblock_estimates,
            black_z,
            white_z,
            is_black,
            eps_minus_phis,
            depth,
        ) = process_frame(
            frame, gray_mapping, os.path.join(work_dir, f"{idx:03}"), data
        )
        frame_stats.append(
            dict(
                errors=errors,
                perblock_estimates=perblock_estimates,
                black_z=black_z,
                white_z=white_z,
                is_black=is_black,
                eps_minus_phis=eps_minus_phis,
                depth=depth,
            )
        )
    generate_frame_stats_distribution(frame_stats, work_dir)
    with open(os.path.join(work_dir, "stats.pkl"), "wb") as f:
        pickle.dump(frame_stats, f)


if __name__ == "__main__":
    main()
