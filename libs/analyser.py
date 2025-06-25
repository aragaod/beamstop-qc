#!/usr/bin/env python3

# Copyright 2025 David Aragao, Diamond Light Source
# Developed with assistance from Google's Gemini.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A library containing the MontageGenerator class for analysing MX diffraction images.

This module provides the core functionality to read HDF5 files from diffraction
experiments, process the data, and generate various analytical plots. It is
designed to be imported into other applications, either for command-line use
or as part of a larger data processing pipeline.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import click
import hdf5plugin
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm

def get_scalar(h5_dataset):
    """
    Reads a scalar HDF5 dataset and returns a single numerical value.

    This helper function correctly handles datasets that are true scalars
    or single-element arrays.

    Args:
        h5_dataset (h5py.Dataset): The HDF5 dataset object to read from.

    Returns:
        A single number (int or float).
    """
    value = h5_dataset[()]
    if isinstance(value, np.ndarray):
        return value.item()
    return value


class MontageGenerator:
    """
    A class to handle the analysis and plotting of a diffraction image montage.

    This class encapsulates all the logic for loading data from a master HDF5 file,
    calculating relevant parameters like resolution, and generating analytical plots
    such as 1D scattering profiles and 2D beam center views.
    """
    def __init__(self, master_filepath):
        """
        Initializes the generator and loads all necessary data from the HDF5 file.

        Args:
            master_filepath (str): The full path to the HDF5 master file.
        """
        if not os.path.exists(master_filepath):
            raise FileNotFoundError(f"The specified file does not exist: {master_filepath}")

        self.master_filepath = master_filepath
        self.beam_info = {}
        self.constants = {}
        self.image_array = None
        self._load_data()

    def _load_data(self):
        """
        Private method to read all required data and metadata from the HDF5 file.

        This method populates the instance's a atributes with data like beam center,
        detector distance, and the main image array. It also robustly determines
        the pixel size and the correct sentinel value for detector gaps.
        """
        click.echo(f"Processing file: {self.master_filepath}")
        PATHS = {
            "data": "/entry/data/data", "beam_center_x": "/entry/instrument/detector/beam_center_x",
            "beam_center_y": "/entry/instrument/detector/beam_center_y",
            "det_distance": "/entry/instrument/detector_z/det_z",
            "wavelength": "/entry/instrument/beam/incident_wavelength",
            "pixel_size_y": "/entry/instrument/detector/y_pixel_size",
        }

        with h5py.File(self.master_filepath, 'r') as hf:
            self.beam_info = {
                'beam_x_px': get_scalar(hf[PATHS["beam_center_x"]]),
                'beam_y_px': get_scalar(hf[PATHS["beam_center_y"]]),
            }
            self.beam_info['beam_x_int'] = int(round(self.beam_info['beam_x_px']))
            self.beam_info['beam_y_int'] = int(round(self.beam_info['beam_y_px']))

            # Robustly read the pixel size, with validation and fallback
            try:
                pixel_size_m = get_scalar(hf[PATHS["pixel_size_y"]])
                pixel_size_mm = pixel_size_m * 1000 # Convert m to mm
                click.echo(f"Read pixel size from file: {pixel_size_m} m -> {pixel_size_mm:.3f} mm")
            except KeyError:
                pixel_size_mm = 0.075 # Fallback to default
                click.secho(f"Warning: Pixel size not found in HDF5 file. Falling back to default: {pixel_size_mm} mm", fg='yellow')

            self.constants = {
                'det_dist_mm': get_scalar(hf[PATHS["det_distance"]]),
                'wavelength_a': get_scalar(hf[PATHS["wavelength"]]),
                'PIXEL_SIZE_MM': pixel_size_mm,
            }
            self.image_array = hf[PATHS["data"]][0, :, :]

        if self.image_array.dtype == np.int32:
            self.constants['GAP_SENTINEL'] = 2**31 - 1
        elif self.image_array.dtype == np.int16:
            self.constants['GAP_SENTINEL'] = 2**15 - 1
        else:
            self.constants['GAP_SENTINEL'] = -1


    def plot_1d_profile(self, ax, slice_type, direction='default'):
        """
        Generates a 1D scattering profile with a dual axis on a given matplotlib axis object.

        The primary axis shows pixel distance from the beam center, while the secondary
        top axis shows the corresponding resolution in Angstroms.

        Args:
            ax (matplotlib.axes.Axes): The axis object on which to draw the plot.
            slice_type (str): The direction of the profile ('vertical', 'horizontal', 'diagonal').
            direction (str, optional): For vertical/horizontal, the direction of the slice.
                                      'up'/'down' for vertical, 'left'/'right' for horizontal.
                                      Defaults to 'default' ('down' and 'right').
        """
        beam_x_px, beam_y_px = self.beam_info['beam_x_px'], self.beam_info['beam_y_px']
        beam_x_int, beam_y_int = self.beam_info['beam_x_int'], self.beam_info['beam_y_int']
        det_dist_mm, wavelength_a = self.constants['det_dist_mm'], self.constants['wavelength_a']
        PIXEL_SIZE_MM, GAP_SENTINEL = self.constants['PIXEL_SIZE_MM'], self.constants['GAP_SENTINEL']

        if slice_type == 'vertical':
            primary_xlabel = "Vertical Distance from Beam Center (pixels)"
            if direction == 'up':
                title = "Vertical Profile (Upwards)"
                profile_slice = np.flip(self.image_array[:beam_y_int, beam_x_int])
                y_coords_abs = np.arange(beam_y_int - 1, -1, -1)
                x_coords_abs = np.full_like(y_coords_abs, beam_x_int, dtype=float)
            else:
                title = "Vertical Profile (Downwards)"
                profile_slice = self.image_array[beam_y_int:, beam_x_int]
                y_coords_abs = np.arange(beam_y_int, beam_y_int + len(profile_slice))
                x_coords_abs = np.full_like(y_coords_abs, beam_x_int, dtype=float)
            primary_axis_coords = np.arange(len(profile_slice))

        elif slice_type == 'horizontal':
            primary_xlabel = "Horizontal Distance from Beam Center (pixels)"
            if direction == 'left':
                title = "Horizontal Profile (Leftwards)"
                profile_slice = np.flip(self.image_array[beam_y_int, :beam_x_int])
                x_coords_abs = np.arange(beam_x_int - 1, -1, -1)
                y_coords_abs = np.full_like(x_coords_abs, beam_y_int, dtype=float)
            else:
                title = "Horizontal Profile (Rightwards)"
                profile_slice = self.image_array[beam_y_int, beam_x_int:]
                x_coords_abs = np.arange(beam_x_int, beam_x_int + len(profile_slice))
                y_coords_abs = np.full_like(x_coords_abs, beam_y_int, dtype=float)
            primary_axis_coords = np.arange(len(profile_slice))
            
        elif slice_type == 'diagonal':
            title = "Diagonal Profile"
            primary_xlabel = "Diagonal Distance from Beam Center (pixels)"
            end_x, end_y = self.image_array.shape[1] - 1, self.image_array.shape[0] - 1
            num_points = int(np.hypot(end_x - beam_x_int, end_y - beam_y_int))
            x_coords_abs = np.linspace(beam_x_int, end_x, num_points)
            y_coords_abs = np.linspace(beam_y_int, end_y, num_points)
            profile_slice = self.image_array[y_coords_abs.astype(int), x_coords_abs.astype(int)]
            primary_axis_coords = np.arange(len(profile_slice))

        x_dist_from_beam_px = x_coords_abs - beam_x_px
        y_dist_from_beam_px = y_coords_abs - beam_y_px
        radial_dist_px = np.sqrt(x_dist_from_beam_px**2 + y_dist_from_beam_px**2)
        r_mm = radial_dist_px * PIXEL_SIZE_MM
        resolution_a = np.full_like(r_mm, np.inf, dtype=float)
        valid_mask = r_mm > 0
        theta = np.full_like(r_mm, 0.0)
        theta[valid_mask] = np.arctan(r_mm[valid_mask] / det_dist_mm) / 2.0
        sintheta = np.sin(theta)
        valid_mask &= (sintheta > 0)
        resolution_a[valid_mask] = wavelength_a / (2 * sintheta[valid_mask])

        valid_interp_mask = np.isfinite(resolution_a)
        interp_pixels = primary_axis_coords[valid_interp_mask]
        interp_res = resolution_a[valid_interp_mask]
        sort_order = np.argsort(interp_res)
        interp_res_sorted = interp_res[sort_order]
        interp_pixels_sorted = interp_pixels[sort_order]

        valid_data_mask = profile_slice < GAP_SENTINEL
        ax.plot(primary_axis_coords[valid_data_mask], profile_slice[valid_data_mask])
        ax.set(xlabel=primary_xlabel, ylabel="Intensity", title=title, yscale='log', xlim=(np.min(primary_axis_coords), np.max(primary_axis_coords)))
        
        secax = ax.secondary_xaxis('top', functions=(
            lambda p: np.interp(p, interp_pixels, interp_res, left=np.nan, right=np.nan),
            lambda r: np.interp(r, interp_res_sorted, interp_pixels_sorted, left=np.nan, right=np.nan)
        ))
        secax.set(xlabel="Resolution (d-spacing) [Å]", xscale='log')
        secax.invert_xaxis()
        
        gap_indices = np.where(profile_slice >= GAP_SENTINEL)[0]
        if gap_indices.size > 0:
            contiguous_gaps = np.split(gap_indices, np.where(np.diff(gap_indices) != 1)[0] + 1)
            for i, gap_group in enumerate(contiguous_gaps):
                ax.axvspan(gap_group[0], gap_group[-1], color='gray', alpha=0.3, zorder=0, label="Detector Gaps" if i == 0 else None)
        ax.grid(True, which='both', linestyle='--')
        if gap_indices.size > 0: ax.legend(loc='best')

    def plot_2d_center(self, ax, cmd_options):
        """Generates a 2D heatmap of the beam center."""
        beam_x_px, beam_y_px = self.beam_info['beam_x_px'], self.beam_info['beam_y_px']
        GAP_SENTINEL = self.constants['GAP_SENTINEL']
        
        box_width, box_height = cmd_options['box_width'], cmd_options['box_height']
        low_intensity_threshold = cmd_options['threshold']
        cmap_name = cmd_options['cmap']
        
        box_half_width, box_half_height = box_width // 2, box_height // 2
        beam_x_int, beam_y_int = self.beam_info['beam_x_int'], self.beam_info['beam_y_int']
        y_start, y_end = beam_y_int - box_half_height, beam_y_int + box_half_height
        x_start, x_end = beam_x_int - box_half_width, beam_x_int + box_half_width
        roi_data = self.image_array[y_start:y_end, x_start:x_end]

        roi_data_filtered = roi_data.astype(float)
        roi_data_filtered[(roi_data_filtered >= GAP_SENTINEL) | (roi_data_filtered <= low_intensity_threshold)] = np.nan
        
        cmap = plt.get_cmap(cmap_name)
        cmap.set_bad(color='white')
        im = ax.imshow(roi_data_filtered, cmap=cmap, norm=LogNorm(vmin=low_intensity_threshold + 1))
        
        center_x_in_roi, center_y_in_roi = beam_x_px - x_start, beam_y_px - y_start
        ax.plot(center_x_in_roi, center_y_in_roi, 'r+', markersize=12, label='Beam Center')
        
        ax.set(xlabel="X Pixel (relative)", ylabel="Y Pixel (relative)", title=f"Beam Center Region ({box_width}x{box_height})")
        ax.legend(loc='lower left')
        return im

    def create_montage(self, cli_options):
        """The main method to orchestrate creating the full 4-panel montage plot."""
        click.echo("Generating montage...")
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        
        fig.suptitle("Diffraction Image Analysis Montage", fontsize=20)
        fig.text(0.5, 0.96, self.master_filepath, ha='center', va='top', fontsize=12, color='gray')
        
        vertical_direction = 'up' if cli_options['vertical_up'] else 'down'
        horizontal_direction = 'left' if cli_options['horizontal_left'] else 'right'

        im2d = self.plot_2d_center(axes[0, 0], cli_options)
        self.plot_1d_profile(axes[0, 1], 'diagonal')
        self.plot_1d_profile(axes[1, 0], 'horizontal', direction=horizontal_direction)
        self.plot_1d_profile(axes[1, 1], 'vertical', direction=vertical_direction)
        
        fig.colorbar(im2d, ax=axes[0, 0], fraction=0.046, pad=0.04, label="Intensity")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        return fig

