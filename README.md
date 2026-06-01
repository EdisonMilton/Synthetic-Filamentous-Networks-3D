## Dataset samples

Due to the large size of the generated volumetric data, the complete synthetic dataset is not included in this repository.

The original dataset used in the study consisted of 50 synthetic 3D filamentous networks with a size of 512 × 512 × 256 voxels.

Each generated sample contains:

- degraded volume (.npy)
- ground truth binary mask (.npy)
- skeleton representation
- graph information

The complete dataset can be regenerated using the script:

src/generacion_datos/dataset_inicial.py

using the same parameters reported in the manuscript.