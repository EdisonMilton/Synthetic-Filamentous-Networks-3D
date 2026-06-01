import os
import numpy as np
import matplotlib.pyplot as plt

# =========================================
# DEPENDENCIA OPCIONAL PARA ESQUELETIZACIÓN
# =========================================
SKIMAGE_OK = False
SKELETON_MODE = None

try:
    from skimage.morphology import skeletonize_3d
    SKIMAGE_OK = True
    SKELETON_MODE = "skeletonize_3d"
except Exception:
    try:
        from skimage.morphology import skeletonize
        SKIMAGE_OK = True
        SKELETON_MODE = "skeletonize_2d_slice_by_slice"
    except Exception:
        SKIMAGE_OK = False
        SKELETON_MODE = None

# =========================================
# CONFIGURACIÓN
# =========================================
DATASET_DIR = r"samples"
SAMPLE = "sample_014"
SAVE_FIGURE = True
OUTPUT_NAME = "figura3_pipeline.png"
DPI = 300

sample_dir = os.path.join(DATASET_DIR, SAMPLE)

# =========================================
# CARGA DE DATOS
# =========================================
volume = np.load(os.path.join(sample_dir, "volume.npy")).astype(np.float32)
frangi_response = np.load(os.path.join(sample_dir, "frangi_response.npy")).astype(np.float32)
mask = np.load(os.path.join(sample_dir, "mask_clean_frangi.npy")) > 0

# =========================================
# ESQUELETO RECONSTRUIDO
# =========================================
if not SKIMAGE_OK:
    raise RuntimeError(
        "No se encontró scikit-image con soporte de skeletonización. "
        "Instala scikit-image para generar el esqueleto."
    )

if SKELETON_MODE == "skeletonize_3d":
    skeleton = skeletonize_3d(mask.astype(np.uint8)) > 0
elif SKELETON_MODE == "skeletonize_2d_slice_by_slice":
    skeleton = np.zeros_like(mask, dtype=bool)
    for z in range(mask.shape[2]):
        skeleton[:, :, z] = skeletonize(mask[:, :, z].astype(np.uint8)) > 0
else:
    raise RuntimeError("No hay método de esqueletización disponible.")

# =========================================
# PROYECCIONES MÁXIMAS (MIP)
# =========================================
vol_mip = np.max(volume, axis=2)
frangi_mip = np.max(frangi_response, axis=2)
mask_mip = np.max(mask.astype(np.uint8), axis=2)
skeleton_mip = np.max(skeleton.astype(np.uint8), axis=2)

# normalización robusta para visualización
def normalize_for_display(img):
    p2, p98 = np.percentile(img, [2, 98])
    if p98 > p2:
        return np.clip((img - p2) / (p98 - p2), 0, 1)
    return np.zeros_like(img, dtype=np.float32)

vol_show = normalize_for_display(vol_mip)
frangi_show = normalize_for_display(frangi_mip)

# =========================================
# FIGURA
# =========================================
fig, ax = plt.subplots(2, 2, figsize=(9, 8), facecolor="white")

ax[0, 0].imshow(vol_show, cmap="gray")
ax[0, 0].set_title("(a) Volumen degradado (MIP)")
ax[0, 0].axis("off")

ax[0, 1].imshow(frangi_show, cmap="magma")
ax[0, 1].set_title("(b) Respuesta Frangi (MIP)")
ax[0, 1].axis("off")

ax[1, 0].imshow(mask_mip, cmap="gray")
ax[1, 0].set_title("(c) Segmentación binaria (MIP)")
ax[1, 0].axis("off")

#ax[1, 1].imshow(skeleton_mip, cmap="gray")
ax[1, 1].imshow(skeleton_mip, cmap="gray", vmin=0, vmax=1)
ax[1, 1].set_title("(d) Esqueleto reconstruido (MIP)")
ax[1, 1].axis("off")

for a in ax.flat:
    a.set_aspect("equal")

plt.tight_layout()

if SAVE_FIGURE:
    output_path = os.path.join(DATASET_DIR, OUTPUT_NAME)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"Figura guardada en: {output_path}")

plt.show()