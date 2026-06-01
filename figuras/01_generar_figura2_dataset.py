#04_generar_figura2_dataset.py
import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = r"samples"
SAMPLE = "sample_005"

sample_dir = os.path.join(BASE_DIR, SAMPLE)

gt_mask = np.load(os.path.join(sample_dir, "gt_mask.npy"))
gt_skeleton = np.load(os.path.join(sample_dir, "gt_skeleton.npy"))
volume = np.load(os.path.join(sample_dir, "volume.npy"))

# PROYECCIÓN MÁXIMA EN Z
mask_mip = np.max(gt_mask, axis=2)
skeleton_mip = np.max(gt_skeleton, axis=2)
volume_mip = np.max(volume, axis=2)

# normalizar volumen para visualizar
p2, p98 = np.percentile(volume_mip, [2, 98])
volume_show = np.clip((volume_mip - p2) / (p98 - p2), 0, 1)

fig, ax = plt.subplots(2, 2, figsize=(10, 8), facecolor="white")

ax[0, 0].imshow(mask_mip, cmap="gray")
ax[0, 0].set_title("(a) Máscara binaria de referencia")
ax[0, 0].axis("off")

ax[0, 1].imshow(skeleton_mip, cmap="gray")
ax[0, 1].set_title("(b) Esqueleto tridimensional de referencia")
ax[0, 1].axis("off")

ax[1, 0].imshow(volume_show, cmap="gray")
ax[1, 0].set_title("(c) Volumen degradado simulado")
ax[1, 0].axis("off")

ax[1, 1].imshow(volume_show, cmap="gray")
ax[1, 1].imshow(mask_mip, cmap="inferno", alpha=0.35)
ax[1, 1].set_title("(d) Superposición entre volumen degradado y referencia")
ax[1, 1].axis("off")

for a in ax.flat:
    a.set_aspect('equal')

plt.tight_layout()
#plt.savefig("figura2_dataset_mip.png", dpi=300)
plt.savefig("figura2_dataset_mip.png", dpi=300, bbox_inches="tight")
plt.show()