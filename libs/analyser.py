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
designed to be imported into other applications or used by the main script.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import hdf5plugin
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm

def get_scalar(h5_dataset):
    """
    Reads a scalar HDF5 dataset and returns a single numerical value.

    This helper function handles the case where a dataset might be a 0-dimensional
    numpy array instead of a true scalar.

    Args:
        h5_dataset (h5py.Dataset): The HDF5 dataset to read.

    Returns:
        A single numerical value (int, float, etc.).
    """
    value = h5_dataset[()]
    if isinstance(value, np.ndarray):
        return value.item()
    return value


class MontageGenerator:
    """
    A class to handle the analysis and plotting of a diffraction image montage.
    """
    def __init__(self, master_filepath):
        """
        Initializes the generator by loading and processing data from the HDF5 file.

        Args:
            master_filepath (str): The path to the HDF5 master file.
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
        Private method to read metadata and image data from the HDF5 master file.
        """
        logging.info(f"Processing file: {self.master_filepath}")

        # --- HDF5/NeXus Data Paths ---
        # This dictionary contains the standard NeXus paths for essential metadata.
        # If using data from a different beamline or facility, you may need to
        # update these paths to match the specific HDF5 file structure.
        PATHS = {
            "data": "/entry/data/data",
            "beam_center_x": "/entry/instrument/detector/beam_center_x",
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

            try:
                pixel_size_m = get_scalar(hf[PATHS["pixel_size_y"]])
                pixel_size_mm = pixel_size_m * 1000
                logging.debug(f"Read pixel size from file: {pixel_size_m} m -> {pixel_size_mm:.3f} mm")
            except KeyError:
                pixel_size_mm = 0.075
                logging.warning(f"Pixel size not found in HDF5 file. Falling back to default: {pixel_size_mm} mm")

            self.constants = {
                'det_dist_mm': get_scalar(hf[PATHS["det_distance"]]),
                'wavelength_a': get_scalar(hf[PATHS["wavelength"]]),
                'PIXEL_SIZE_MM': pixel_size_mm,
            }
            self.image_array = hf[PATHS["data"]][0, :, :]

        # --- Dynamic Gap Sentinel Detection ---
        # Detectors like Pilatus and Eiger use a large positive integer to flag
        # pixels in physical gaps between modules. This value depends on the
        # bit-depth of the data. We dynamically check the data type to use the
        # correct sentinel value, making the script more robust.
        if self.image_array.dtype == np.int32:
            self.constants['GAP_SENTINEL'] = 2**31 - 1
        elif self.image_array.dtype == np.int16:
            self.constants['GAP_SENTINEL'] = 2**15 - 1
        else:
            # Fallback for other data types where -1 is a common mask value.
            self.constants['GAP_SENTINEL'] = -1

    def plot_1d_profile(self, ax, slice_type, direction='default'):
        """
        Generates a 1D scattering profile with manually overlaid resolution ticks.

        This method plots intensity versus radial distance from the beam center.
        It then calculates the corresponding resolution (d-spacing) for a
        pre-defined set of ticks and manually draws them with labels and gridlines
        at the top of the plot, creating a "fake" secondary axis.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis object to plot on.
            slice_type (str): The type of profile ('vertical', 'horizontal', 'diagonal').
            direction (str, optional): The direction for vertical/horizontal profiles.
        """
        beam_x_px, beam_y_px = self.beam_info['beam_x_px'], self.beam_info['beam_y_px']
        beam_x_int, beam_y_int = self.beam_info['beam_x_int'], self.beam_info['beam_y_int']
        det_dist_mm, wavelength_a = self.constants['det_dist_mm'], self.constants['wavelength_a']
        PIXEL_SIZE_MM, GAP_SENTINEL = self.constants['PIXEL_SIZE_MM'], self.constants['GAP_SENTINEL']

        if slice_type == 'vertical':
            if direction == 'up':
                title = "Vertical Profile (Upwards)"
                profile_slice = np.flip(self.image_array[:beam_y_int, beam_x_int])
                y_coords_abs = np.arange(beam_y_int - 1, -1, -1)
            else:
                title = "Vertical Profile (Downwards)"
                profile_slice = self.image_array[beam_y_int:, beam_x_int]
                y_coords_abs = np.arange(beam_y_int, beam_y_int + len(profile_slice))
            x_dist_px = beam_x_int - beam_x_px
            y_dist_px = y_coords_abs - beam_y_px
            radial_dist_px = np.sqrt(x_dist_px**2 + y_dist_px**2)

        elif slice_type == 'horizontal':
            if direction == 'left':
                title = "Horizontal Profile (Leftwards)"
                profile_slice = np.flip(self.image_array[beam_y_int, :beam_x_int])
                x_coords_abs = np.arange(beam_x_int - 1, -1, -1)
            else:
                title = "Horizontal Profile (Rightwards)"
                profile_slice = self.image_array[beam_y_int, beam_x_int:]
                x_coords_abs = np.arange(beam_x_int, beam_x_int + len(profile_slice))
            x_dist_px = x_coords_abs - beam_x_px
            y_dist_px = beam_y_int - beam_y_px
            radial_dist_px = np.sqrt(x_dist_px**2 + y_dist_px**2)

        elif slice_type == 'diagonal':
            title = "Diagonal Profile"
            end_x, end_y = self.image_array.shape[1] - 1, self.image_array.shape[0] - 1
            num_points = int(np.hypot(end_x - beam_x_int, end_y - beam_y_int))
            x_coords_abs = np.linspace(beam_x_int, end_x, num_points)
            y_coords_abs = np.linspace(beam_y_int, end_y, num_points)
            profile_slice = self.image_array[y_coords_abs.astype(int), x_coords_abs.astype(int)]
            x_dist_px = x_coords_abs - beam_x_px
            y_dist_px = y_coords_abs - beam_y_px
            radial_dist_px = np.sqrt(x_dist_px**2 + y_dist_px**2)

        def resolution_to_pixel(res):
            """
            Converts resolution in Angstroms to radial distance in pixels.
            This calculation uses Bragg's Law. First, the scattering angle (theta)
            is determined from the resolution (d) and wavelength (lambda) using
            theta = arcsin(lambda / 2d). Then, simple trigonometry gives the
            radial distance on the detector in mm, which is converted to pixels.
            """
            with np.errstate(divide='ignore', invalid='ignore'):
                sintheta = np.divide(wavelength_a, (2 * res), where=(res > 0))

                if isinstance(sintheta, np.ndarray):
                    sintheta[sintheta > 1] = np.nan
                elif sintheta > 1:
                    sintheta = np.nan

                theta = np.arcsin(sintheta)
                r_mm = det_dist_mm * np.tan(2 * theta)
                r_px = r_mm / PIXEL_SIZE_MM

                if isinstance(r_px, np.ndarray):
                    r_px[np.isnan(r_px)] = -1
                elif np.isnan(r_px):
                    r_px = -1
            return r_px

        # Calculate the true radial distance of each point in the slice from
        # the (potentially fractional) beam center using the Pythagorean theorem.
        primary_axis_coords = radial_dist_px
        valid_data_mask = profile_slice < GAP_SENTINEL
        ax.plot(primary_axis_coords[valid_data_mask], profile_slice[valid_data_mask])

        # Format the primary plot, removing automatic title placement
        ax.set(xlabel="Radial Distance from Beam Center (pixels)", ylabel="Intensity", yscale='log')
        ax.grid(True, which='both', linestyle='--')

        # Create ALL titles manually to control order and position
        ax.text(0.5, 1.10, title, ha='center', va='bottom', transform=ax.transAxes, fontsize=12, weight='bold')
        
        # --- The "Fake" Secondary Axis Hack ---
        # Right then, listen up. You might be wondering why we're not using Matplotlib's
        # fancy `secondary_xaxis`. The simple answer: it's gone completely walkabout.
        # Trying to get an inverted, logarithmic secondary axis with fixed ticks to work
        # reliably was a proper dog's breakfast.
        #
        # So, instead of fighting the beast, we're giving this manual overlay a burl.
        # We plot against the simple, linear pixel axis and then go back and draw our
        # own resolution ticks and labels exactly where we want them. It's a bit of a
        # hack, but she'll be right, mate. It gives us total control and a beaut plot.
        ax.text(0.5, 1.05, "Resolution (d-spacing) [Å]", ha='center', va='bottom', transform=ax.transAxes)

        resolution_ticks = [100, 10, 3.5, 2.0, 1.5, 1.0]
        xmin, xmax = ax.get_xlim()

        logging.debug(f"--- Debugging axis for: {title} ---")
        logging.debug(f"Primary axis (pixel) limits are: xmin={xmin:.2f}, xmax={xmax:.2f}")

        for res_value in resolution_ticks:
            pixel_pos = resolution_to_pixel(res_value)
            
            if xmin <= pixel_pos <= xmax:
                logging.debug(f"  -> Resolution tick {res_value} Å is visible at pixel position {pixel_pos:.2f}")
                ax.axvline(x=pixel_pos, color='gray', linestyle=':', linewidth=1, zorder=0)

                label = f"{res_value:.1f}"
                if res_value >= 10:
                    label = f"{res_value:.0f}"

                bbox_props = dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='none')
                ax.text(pixel_pos, 0.98, label, ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=8, bbox=bbox_props)
            else:
                logging.debug(f"  -> Resolution tick {res_value} Å (at pixel {pixel_pos:.2f}) is outside plot range.")

        # Visualize detector gaps using pixel coordinates
        gap_indices = np.where(profile_slice >= GAP_SENTINEL)[0]
        if gap_indices.size > 0:
            gap_pixel_coords = primary_axis_coords[gap_indices]
            contiguous_gaps = np.split(gap_pixel_coords, np.where(np.diff(gap_pixel_coords) > 1.5)[0] + 1)
            for i, gap_group in enumerate(contiguous_gaps):
                label = "Detector Gaps" if i == 0 else None
                ax.axvspan(xmin=gap_group[0], xmax=gap_group[-1], color='gray', alpha=0.3, zorder=0, label=label)

            ax.legend(loc='center right')

    def plot_2d_center(self, ax, cmd_options):
        """
        Generates a 2D heatmap of the region around the beam center.

        This plot shows a log-scale intensity heatmap of a square region centered
        on the beam position. It includes a marker for the precise beam center and
        can overlay binned intensity values for quantitative analysis.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis object to plot on.
            cmd_options (dict): A dictionary of command-line options.
        
        Returns:
            matplotlib.image.AxesImage: The image object for the colorbar.
        """
        beam_x_px, beam_y_px = self.beam_info['beam_x_px'], self.beam_info['beam_y_px']
        GAP_SENTINEL = self.constants['GAP_SENTINEL']

        box_width, box_height = cmd_options['box_width'], cmd_options['box_height']
        low_intensity_threshold = cmd_options['threshold']
        annotation_threshold = cmd_options['annotation_threshold']
        cmap_name = cmd_options['cmap']
        bin_size = cmd_options['bin_size']

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

        # Add binned intensity annotations
        for r in range(0, roi_data.shape[0], bin_size):
            for c in range(0, roi_data.shape[1], bin_size):
                block = roi_data[r:r+bin_size, c:c+bin_size]
                valid_pixels = block[block < GAP_SENTINEL]
                if valid_pixels.size > 0:
                    mean_intensity = np.mean(valid_pixels)
                    if mean_intensity > annotation_threshold:
                        bbox_props = dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5, edgecolor='none')
                        ax.text(c + bin_size/2, r + bin_size/2, f"{mean_intensity:.0f}",
                                ha="center", va="center", color="white", fontsize=6, bbox=bbox_props)

        center_x_in_roi, center_y_in_roi = beam_x_px - x_start, beam_y_px - y_start
        ax.plot(center_x_in_roi, center_y_in_roi, 'r+', markersize=12, label='Beam Center')
        
        # Apply consistent manual title styling
        title = f"Beam Center Region ({box_width}x{box_height})"
        ax.set(xlabel="X Pixel (relative)", ylabel="Y Pixel (relative)")
        ax.text(0.5, 1.10, title, ha='center', va='bottom', transform=ax.transAxes, fontsize=12, weight='bold')

        ax.legend(loc='lower left')
        return im

    def create_montage(self, cli_options):
        """
        Orchestrates the creation of the entire figure, which can be a single
        plot, a two-panel plot, or the full four-panel montage.

        Args:
            cli_options (dict): A dictionary of command-line options.

        Returns:
            matplotlib.figure.Figure: The complete figure object.
        """
        logging.info("Generating montage...")

        montage_type = cli_options['montage_type']

        if montage_type == 'full':
            fig, axes = plt.subplots(2, 2, figsize=(18, 16))
            fig.suptitle("Diffraction Image Analysis Montage", fontsize=20)
            
            vertical_direction = 'up' if cli_options['vertical_up'] else 'down'
            horizontal_direction = 'left' if cli_options['horizontal_left'] else 'right'

            im2d = self.plot_2d_center(axes[0, 0], cli_options)
            self.plot_1d_profile(axes[0, 1], 'diagonal')
            self.plot_1d_profile(axes[1, 0], 'horizontal', direction=horizontal_direction)
            self.plot_1d_profile(axes[1, 1], 'vertical', direction=vertical_direction)
            
            fig.colorbar(im2d, ax=axes[0, 0], fraction=0.046, pad=0.04, label="Intensity")

        else: # Single or two-panel montages
            if montage_type in ['vertical', 'horizontal', 'diagonal']:
                fig, axes = plt.subplots(1, 2, figsize=(18, 8))
                fig.suptitle(f"Diffraction Image Analysis: 2D Center and {montage_type.capitalize()} Profile", fontsize=16)
                
                im2d = self.plot_2d_center(axes[0], cli_options)
                fig.colorbar(im2d, ax=axes[0], fraction=0.046, pad=0.04, label="Intensity")
                
                if montage_type == 'vertical':
                    direction = 'up' if cli_options['vertical_up'] else 'down'
                    self.plot_1d_profile(axes[1], 'vertical', direction=direction)
                elif montage_type == 'horizontal':
                    direction = 'left' if cli_options['horizontal_left'] else 'right'
                    self.plot_1d_profile(axes[1], 'horizontal', direction=direction)
                elif montage_type == 'diagonal':
                    self.plot_1d_profile(axes[1], 'diagonal')
            else: # Just a single plot
                fig, ax = plt.subplots(figsize=(10,10))
                if montage_type == '2d':
                    im = self.plot_2d_center(ax, cli_options)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Intensity")
                elif montage_type == '1d_vertical':
                    direction = 'up' if cli_options['vertical_up'] else 'down'
                    self.plot_1d_profile(ax, 'vertical', direction=direction)
                elif montage_type == '1d_horizontal':
                    direction = 'left' if cli_options['horizontal_left'] else 'right'
                    self.plot_1d_profile(ax, 'horizontal', direction=direction)
                elif montage_type == '1d_diagonal':
                     self.plot_1d_profile(ax, 'diagonal')
        
        fig.text(0.5, 0.95, self.master_filepath, ha='center', va='top', fontsize=12, color='gray')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig
