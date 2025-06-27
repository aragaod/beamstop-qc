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
Example script to demonstrate using the MontageGenerator library to create
a single plot of the diagonal scattering profile.
"""

import matplotlib.pyplot as plt
import logging
from pathlib import Path
from libs import MontageGenerator

# --- Configuration ---
MASTER_FILE = Path('data/low_res_100_100_1_master.h5')
OUTPUT_DIR = Path('output_examples')
OUTPUT_FILE = OUTPUT_DIR / 'diagonal_profile_example.png'

def main():
    """
    Main function to run the example.
    """
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logging.info("--- Running Diagonal Profile Example ---")
    
    try:
        analyser = MontageGenerator(MASTER_FILE)
    except FileNotFoundError:
        logging.error(f"Data file not found at '{MASTER_FILE}'.")
        logging.error("Please download the example dataset and place it in the 'data' directory.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    # Correctly call the plot_1d_profile method for a diagonal slice
    analyser.plot_1d_profile(ax, slice_type='diagonal')
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    logging.info(f"Saving plot to {OUTPUT_FILE}...")
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    plt.show()
    logging.info("--- Example Finished ---")

if __name__ == '__main__':
    main()
