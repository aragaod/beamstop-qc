# Diffraction Image Analysis Tool

A Python-based application for generating diagnostic plots from MX diffraction images. This tool is designed to provide a quick quality control check, particularly for verifying beamstop alignment after changes in experimental setup (e.g., switching between low, standard, and high-resolution configurations).

This project was developed as a module for the [mx_beamline_setup](https://github.com/aragaod/mx_beamline_setup) quality control suite at the I04 beamline, Diamond Light Source, but is designed to be a standalone tool that can be adapted for use at other synchrotron beamlines.

![Example Montage](./output_examples/montage_example.png)

---

## Key Features

-   **Montage Generation:** Creates a 2x2 montage including a 2D view of the beam center and 1D scattering profiles in vertical, horizontal, and diagonal directions.
-   **Flexible Layouts:** Generate the full 4-panel montage, a single plot (2D or a specific 1D profile), or a 2-panel view combining the beam center with any 1D profile.
-   **Command-Line Interface:** A flexible CLI powered by `click` allows for easy customisation of plot parameters, input/output files, and profile directions.
-   **Library Usage:** The core logic is encapsulated in a `MontageGenerator` class, allowing it to be easily imported and integrated into larger data processing or analysis pipelines.
-   **Robust Data Handling:** Automatically detects detector bit depth (16/32-bit) to correctly identify detector gaps and handles unreliable metadata by using known physical constants.

---

## Installation & Setup

### 1. Clone the Repository
Clone or download this repository to your local machine.

### 2. Install Dependencies
It is recommended to use a virtual environment. Install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Download Example Data
This project requires example data files to run. The test datasets are archived on Zenodo.

1.  **Download the data archive** from the following DOI:
    [https://doi.org/10.5281/zenodo.15739324](https://doi.org/10.5281/zenodo.15739324)
2.  **Create a `data` directory** in the root of this project folder.
3.  **Unzip the archive** and place the contents (e.g., `low_res_100_100_1_master.h5` and its associated `_meta.h5` and `_..._000001.h5` files) into the `data` directory you just created.

**Citation for the dataset:**
> Aragao, D., & Diamond Light Source. (2025). Example Diffraction Datasets for Beamstop Alignment QC (Version 1.0) [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.15739324](https://doi.org/10.5281/zenodo.15739324)

---

## Usage

The application is run from the command line, pointing to an HDF5 master file. The `--montage-type` option controls the output layout.

### Generating the Full Montage (Default)
This is the default behavior if no `--montage-type` is specified.
```bash
python3 generate_montage.py data/low_res_100_100_1_master.h5
```

### Generating Specific Plots or Layouts
Use the `--montage-type` option to select a different layout.

```bash
# Generate only the 2D beam center plot
python3 generate_montage.py data/low_res_100_100_1_master.h5 --montage-type 2d

# Generate only the vertical 1D profile, going upwards from the beam center
python3 generate_montage.py data/low_res_100_100_1_master.h5 --montage-type 1d_vertical --vertical-up

# Generate a two-panel plot showing the beam center and the horizontal profile
python3 generate_montage.py data/low_res_100_100_1_master.h5 --montage-type horizontal
```

### Enabling Debug Output
For more verbose output, which is useful for diagnostics, add the `--debug` flag to any command:
```bash
python3 generate_montage.py data/low_res_100_100_1_master.h5 --debug
```

### Getting Help
To see a full list of all available command-line options, use the `--help` flag:
```bash
python3 generate_montage.py --help
```

---

## Using as a Library

The core functionality can be easily integrated into other Python scripts. For practical, working examples, please see the scripts in the `/examples` directory.

A basic example is shown below:
```python
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from libs.analyser import MontageGenerator

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Use pathlib for more robust path handling
master_file = Path('data/low_res_100_100_1_master.h5')

try:
    analyser = MontageGenerator(master_file)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Generate a specific 1D profile
    analyser.plot_1d_profile(ax, slice_type='vertical', direction='down')
    
    plt.savefig('my_vertical_plot.png', dpi=150, bbox_inches='tight')
    plt.show()

except FileNotFoundError:
    logging.error(f"Data file not found at '{master_file}'.")
```

---
## Acknowledgements
Portions of this software were developed with assistance from Google's Gemini.

## License
This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.



