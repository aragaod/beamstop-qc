#!/usr/bin/env python3

"""
A command-line tool to generate analysis plots from an MX diffraction image.

This script acts as the main entry point for the user. It uses the 'click'
library to provide a user-friendly command-line interface and handles the
overall execution flow, including setting up a logger that can be controlled
by a --debug flag for more verbose output.
"""

import click
import matplotlib.pyplot as plt
import logging

# Import the main class from our library file
from libs.analyser import MontageGenerator

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('master_filepath', type=click.Path(exists=True, dir_okay=False))
# --- Control and Output Options ---
@click.option('--output', default='diffraction_montage.png', help='Output filename for the plot.', show_default=True)
@click.option('--show-plot/--no-show-plot', default=False, help='Display the plot interactively after saving.')
@click.option('--debug', is_flag=True, help='Enable debug logging for detailed output.')
# --- Plot Selection ---
@click.option('--montage-type', default='full',
              type=click.Choice(['full', 'vertical', 'horizontal', 'diagonal', '2d', '1d_vertical', '1d_horizontal', '1d_diagonal'], case_sensitive=False),
              help='Type of montage to generate.', show_default=True)
# --- 2D Plot Options ---
@click.option('--cmap', default='Greys_r', help='Colormap for the 2D beam center plot.',
              type=click.Choice(['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys_r'], case_sensitive=False), show_default=True)
@click.option('--threshold', default=8, type=int, help='Low intensity threshold for the 2D beam center plot.', show_default=True)
@click.option('--annotation-threshold', default=75, type=int, help='Display intensity values for box of nxn pixel bins above this average.', show_default=True)
@click.option('--bin-size', default=2, type=int, help='The bin size (n x n pixels) for intensity annotations.', show_default=True)
@click.option('--box-width', default=60, type=int, help='Width of the 2D plot window in pixels.', show_default=True)
@click.option('--box-height', default=70, type=int, help='Height of the 2D plot window in pixels.', show_default=True)
# --- 1D Plot Options ---
@click.option('--vertical-up', is_flag=True, help='Generate vertical profile going upwards from beam center.')
@click.option('--horizontal-left', is_flag=True, help='Generate horizontal profile going left from beam center.')
def main(**cli_options):
    """
    Reads an HDF5 master file from a diffraction experiment, analyzes its contents,
    and generates a montage of analytical plots which are then saved to a file.

    This function orchestrates the entire process:
    1. Sets up application-wide logging based on the --debug flag.
    2. Instantiates the MontageGenerator class from the analyser library.
    3. Calls the appropriate methods to create the plots.
    4. Saves the resulting figure to disk and optionally displays it.
    """
    debug_mode = cli_options.pop('debug')

    # --- Logger Configuration ---
    # A robust setup that configures the root logger. This is more reliable
    # than basicConfig() which can be ignored if a library configures logging first.
    log_level = logging.DEBUG if debug_mode else logging.INFO
    log_format = '[%(levelname)s] %(message)s'

    logger = logging.getLogger()
    logger.setLevel(log_level)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)
    # --- End of Logger Configuration ---

    # Silence matplotlib's overly verbose debug messages
    if debug_mode:
        logging.getLogger('matplotlib').setLevel(logging.INFO)

    try:
        analyser = MontageGenerator(cli_options.pop('master_filepath'))

        fig = analyser.create_montage(cli_options)

        if fig:
            output_path = cli_options['output']
            logging.info(f"Saving plot to: {output_path}")
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            if cli_options.get('show_plot'):
                logging.info("Displaying plot...")
                plt.show()

        logging.info("Done!")

    except Exception as e:
        # exc_info=True will automatically include traceback in debug mode
        logging.error(f"An unexpected error occurred: {e}", exc_info=debug_mode)

if __name__ == '__main__':
    main()
