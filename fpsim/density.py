"""Density calculation for fpsim."""
from __future__ import annotations
import IMP
import IMP.em
import numpy as np
from pathlib import Path

class GetModelDensity:
    """Compute mean density maps from structures."""

    def __init__(self, resolution: float = 10.0, voxel_size: float = 5.0):
        """Constructor.

        Args:
            resolution: The MRC resolution of the output map (in Angstrom).
            voxel_size: The voxel size for the output map (in Angstrom).
        """
        self.resolution = resolution
        self.voxel_size = voxel_size
        self.count_models = 0
        self.density_map = None

    def add_model(self, particles: list[IMP.Particle]):
        """Add a model's particles to the density map."""
        if not particles:
            return

        # Create the density map directly from the provided particles.
        # This is much more efficient as it avoids creating a temporary model and copying particles.
        dmap = IMP.em.SampledDensityMap(particles, self.resolution, self.voxel_size)
        dmap.set_was_used(True)

        if self.density_map is None:
            self.density_map = dmap
        else:
            # Expand bounding box and add densities
            bbox1 = IMP.em.get_bounding_box(self.density_map)
            bbox2 = IMP.em.get_bounding_box(dmap)
            bbox1 += bbox2
            new_dmap = IMP.em.create_density_map(bbox1, self.voxel_size)
            new_dmap.set_was_used(True)
            new_dmap.add(dmap)
            new_dmap.add(self.density_map)
            self.density_map = new_dmap

        self.count_models += 1

    def write_mrc(self, output_path: Path, normalize: bool = True):
        """Write the density map to an MRC file."""
        if self.density_map is None:
            raise ValueError("No models were added to the density map.")

        if normalize and self.count_models > 0:
            self.density_map.multiply(1.0 / self.count_models)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        IMP.em.write_map(self.density_map, str(output_path), IMP.em.MRCReaderWriter())
