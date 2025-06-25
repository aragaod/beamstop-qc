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
a single plot of the vertical scattering profile.
"""

import matplotlib.pyplot as plt

from libs import MontageGenerator

# --- Configuration ---
MASTER_FILE = 'data/low_res_100_100_1_master.h5'
OUTPUT_FILE = 'output_examples/example_vertical_profile.png'

def main():
    """
    Main function to run the example.
    """
    print("--- Running Vertical Profile Example ---")
    
    try:
        analyser = MontageGenerator(MASTER_FILE)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{MASTER_FILE}'.")
        print("Please download the example dataset from Zenodo and place it in the 'data' directory.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    analyser.plot_1d_profile(ax, slice_type='vertical', direction='up')
    
    print(f"Saving plot to {OUTPUT_FILE}...")
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    plt.show()
    print("--- Example Finished ---")

if __name__ == '__main__':
    main()

