#pipeline
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, restoration
from skimage.filters import frangi
from skimage.morphology import ball


def normalize_percentile(volume: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    v = volume.astype(np.float32)
    lo, hi = np.percentile(v, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(v, dtype=np.float32)
    v = np.clip((v - lo) / (hi - lo), 0, 1)
    return v


def denoise_volume(volume: np.ndarray, method: str = "median", median_size: int = 3) -> np.ndarray:
    if method == "median":
        return ndi.median_filter(volume, size=median_size)
    raise ValueError("Por ahora usa method='median'")


def subtract_background(volume: np.ndarray, sigma_bg: float = 6.0) -> np.ndarray:
    bg = ndi.gaussian_filter(volume, sigma=sigma_bg)
    corrected = volume - bg
    corrected[corrected < 0] = 0
    return corrected.astype(np.float32)


def frangi_multiscale(volume: np.ndarray, sigmas=(0.5, 1.0, 1.5, 2.0)) -> np.ndarray:
    vesselness = frangi(
        volume,
        sigmas=sigmas,
        alpha=0.5,
        beta=0.5,
        gamma=None,
        black_ridges=False,
    )
    vesselness = np.nan_to_num(vesselness, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return vesselness


def segment_vessels(
    vesselness: np.ndarray,
    threshold_percentile: float = 99.0,
    min_size: int = 5,
    closing_radius: int = 0,
    hole_area: int = 0,
) -> np.ndarray:
    """
    Segmentación MUY permisiva para depuración.
    """
    th = np.percentile(vesselness, threshold_percentile)
    mask = vesselness > th

    # adelgazar un poco la máscara
    mask = morphology.binary_opening(mask, footprint=ball(1))

    if min_size > 0:
        mask = morphology.remove_small_objects(mask, min_size=min_size)

    if closing_radius > 0:
        mask = morphology.closing(mask, footprint=ball(closing_radius))

    if hole_area > 0:
        mask = morphology.remove_small_holes(mask, area_threshold=hole_area)

    return mask.astype(bool)


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    if hasattr(morphology, "skeletonize_3d"):
        skel = morphology.skeletonize_3d(mask)
    else:
        skel = morphology.skeletonize(mask)
    return skel.astype(bool)


def get_neighbor_count(skel: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    neighbors = ndi.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    return neighbors


def classify_skeleton_points(skel: np.ndarray):
    neighbors = get_neighbor_count(skel)
    endpoints = skel & (neighbors == 1)
    branchpoints = skel & (neighbors >= 3)
    centerline = skel & (neighbors == 2)
    return endpoints, branchpoints, centerline


def remove_points_near_mask(points: np.ndarray, forbidden: np.ndarray, radius: int = 2) -> np.ndarray:
    if radius <= 0:
        return points
    dilated = morphology.dilation(forbidden, footprint=ball(radius))
    return points & (~dilated)


def compute_local_diameter(
    mask: np.ndarray,
    skel: np.ndarray,
    exclude_endpoints_radius: int = 1,
    exclude_branchpoints_radius: int = 1,
):
    dist = ndi.distance_transform_edt(mask)
    diam_map = 2.0 * dist

    endpoints, branchpoints, centerline = classify_skeleton_points(skel)

    valid_points = centerline.copy()
    valid_points = remove_points_near_mask(valid_points, endpoints, radius=exclude_endpoints_radius)
    valid_points = remove_points_near_mask(valid_points, branchpoints, radius=exclude_branchpoints_radius)

    diam_samples = diam_map[valid_points]
    return diam_map, valid_points, diam_samples


def robust_stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan,
        }

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def vessel_pipeline(
    volume: np.ndarray,
    denoise_method: str = "median",
    median_size: int = 3,
    sigma_bg: float = 6.0,
    frangi_sigmas=(0.5, 1.0, 1.5, 2.0),
    threshold_percentile: float = 99.0,
    min_size: int = 5,
    closing_radius: int = 0,
    hole_area: int = 0,
):
    vol_norm = normalize_percentile(volume, p_low=1, p_high=99)
    vol_denoised = denoise_volume(vol_norm, method=denoise_method, median_size=median_size)
    vol_corr = subtract_background(vol_denoised, sigma_bg=sigma_bg)
    vesselness = frangi_multiscale(vol_corr, sigmas=frangi_sigmas)

    mask = segment_vessels(
        vesselness,
        threshold_percentile=threshold_percentile,
        min_size=min_size,
        closing_radius=closing_radius,
        hole_area=hole_area,
    )

    skel = skeletonize_mask(mask)

    diam_map, valid_points, diam_samples = compute_local_diameter(
        mask,
        skel,
        exclude_endpoints_radius=1,
        exclude_branchpoints_radius=1,
    )

    stats = robust_stats(diam_samples)

    return {
        "volume_norm": vol_norm,
        "volume_denoised": vol_denoised,
        "volume_corrected": vol_corr,
        "vesselness": vesselness,
        "mask": mask,
        "skeleton": skel,
        "diameter_map": diam_map,
        "valid_points": valid_points,
        "diameter_samples": diam_samples,
        "diameter_stats": stats,
    }