#!/usr/bin/env python3

# Copyright 2025 David Aragao, Diamond Light Source
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
A command-line tool to generate analysis plots from an MX diffraction image.

This script uses the 'click' library to provide a user-friendly interface for
the MontageGenerator class, allowing for the creation of either a full 4-panel
montage or individual diagnostic plots.
"""

import click
import matplotlib.pyplot as plt
import traceback

# Import the main class from our library file
from libs import MontageGenerator

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('master_filepath', type=click.Path(exists=True, dir_okay=False))
# --- Output Options ---
@click.option('--output', default='diffraction_plot.png', help='Output PNG file path.', show_default=True)
@click.option('--show-plot/--no-show-plot', default=False, help='Display the plot interactively after saving.')
# --- 2D Plot Options ---
@click.option('--cmap', default='viridis', help='Colormap for the 2D beam center plot.',
              type=click.Choice(['viridis', 'plasma', 'inferno', 'magma', 'cividis'], case_sensitive=False), show_default=True)
@click.option('--threshold', default=8, type=int, help='Low intensity threshold for the 2D beam center plot.', show_default=True)
@click.option('--box-width', default=100, type=int, help='Width of the 2D plot window in pixels.', show_default=True)
@click.option('--box-height', default=75, type=int, help='Height of the 2D plot window in pixels.', show_default=True)
# --- 1D Plot Options ---
@click.option('--vertical-up', is_flag=True, help='Generate vertical profile going upwards from beam center.')
@click.option('--horizontal-left', is_flag=True, help='Generate horizontal profile going left from beam center.')
# --- Single Plot Generation ---
@click.option('--plot-2d-only', is_flag=True, help='Only generate the 2D beam center plot.')
@click.option('--plot-vertical-only', is_flag=True, help='Only generate the vertical profile plot.')
@click.option('--plot-horizontal-only', is_flag=True, help='Only generate the horizontal profile plot.')
@click.option('--plot-diagonal-only', is_flag=True, help='Only generate the diagonal profile plot.')

def main(**cli_options):
    """
    Reads an HDF5 master file from a diffraction experiment, analyzes it,
    and generates a 4-panel montage plot or individual plots which are saved to a file.
    """
    try:
        # Create an instance of the analyser class
        analyser = MontageGenerator(cli_options.pop('master_filepath'))

        single_plot_flags = {
            '2d': cli_options.pop('plot_2d_only'),
            'vertical': cli_options.pop('plot_vertical_only'),
            'horizontal': cli_options.pop('plot_horizontal_only'),
            'diagonal': cli_options.pop('plot_diagonal_only'),
        }

        # Check that only one single-plot flag is used
        if sum(single_plot_flags.values()) > 1:
            raise click.UsageError("Please choose only one '--plot-*-only' option at a time.")

        output_path = cli_options['output']
        fig = None # Initialize fig to None

        # Logic to call the correct plotting method based on flags
        if single_plot_flags['2d']:
            fig, ax = plt.subplots(figsize=(10, 10))
            im = analyser.plot_2d_center(ax, cli_options)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Intensity")
        elif single_plot_flags['vertical']:
            fig, ax = plt.subplots(figsize=(12, 7))
            direction = 'up' if cli_options['vertical_up'] else 'down'
            analyser.plot_1d_profile(ax, 'vertical', direction=direction)
        elif single_plot_flags['horizontal']:
            fig, ax = plt.subplots(figsize=(12, 7))
            direction = 'left' if cli_options['horizontal_left'] else 'right'
            analyser.plot_1d_profile(ax, 'horizontal', direction=direction)
        elif single_plot_flags['diagonal']:
            fig, ax = plt.subplots(figsize=(12, 7))
            analyser.plot_1d_profile(ax, 'diagonal')
        else:
            # If no single-plot flag is given, create the full montage
            fig = analyser.create_montage(cli_options)

        # Save the figure if one was created
        if fig:
            click.echo(f"Saving plot to: {output_path}")
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            if cli_options.get('show_plot'):
                click.echo("Displaying plot...")
                plt.show()

        click.secho("Done!", fg='green')

    except Exception as e:
        click.secho(f"An error occurred: {e}", fg='red', err=True)
        traceback.print_exc()

if __name__ == '__main__':
    main()

