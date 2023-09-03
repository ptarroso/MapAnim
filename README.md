# MapAnim

MapAnim is a Python script designed to create captivating visualizations of 
raster maps. It achieves this by animating the movement of data pixels, 
arranging the data in spirals, and visually representing the proportions of
each class.

The script can be run from the command line and is highly customizable using 
simple text files. Included with the script is an illustrative example 
featuring world biomes.

*NOTE*: While the initial intent behind this script was simplicity of the code, 
streamlining its usage introduced some complexity. Due to my lack of time and a
sporadic development effort, the script lacks a structured design which 
resulted in a kind of spaghetti code. Nevertheless, it does accomplishes its 
animating goals (although it was not fully tested)!

## Instalation

There are a few dependencies required to run the script. It necessitates a 
working installation of Python with numpy, matplotlib, and gdal. If you have 
these dependencies installed system-wide, you can use your system's Python.

Additionally, a mapanim.yml file is provided to create a Conda environment with
he required dependencies. To set up the environment, ensure you have Conda 
installed, and then run the following commands:

```bash
conda env create -f mapanim.yml 
```

Activate the environment with:

```bash
conda activate mapanim
```

Now, you can run the script within the mapanim environment.

## Usage

The usage of the **mapAnim.py** script is straightforward. It requires three 
input files:

1.**Raster File**: Ideally in TIF format, this is a categorical map containing
different classes to be animated.

2.**Class Data**: A CSV file that relates values in the raster to descriptions,
color codes, and (x, y) targets for spiral packing.

3.**Elements**: A JSON file containing elements (text, lines, scale bars, north
arrows) and customizations to be plotted alongside the map.

The script generates a sequence of image frames and saves them to a specified 
folder. These frames can be combined into a video file using any video editing 
software. The script offers multiple arguments for fine-tuning the output. For 
example, you can control the output size using the 'size' or '-s' argument and 
adjust the positioning of the raster map within the frame using the '--xlim' or 
'-x' and '--ylim' or '-y' arguments. These axis limits are in pixel units (1 
unit equals 1 pixel) and are relative to the size of the original raster.

Keep in mind that the size of the raster can be limiting depending on available
RAM. To address this, the script provides a resampling option, maintaining 
calculations of proportions with the original raster resolution. For a very 
large raster, you can decrease the resolution by a factor, e.g., '-r 4' for a
factor of 4.

As an example, to generate a 4k output in a 'frame_output' folder for a raster 
requiring a factor 4 resampling, you can run the following command:

```bash
python mapAnim.py -r 4 -s 3840 2160 RASTER.tif CLASSDATA.csv ELEMENTS.json frame_output
```

Numerous other arguments are available for controlling animation properties.
You can check the available options with:

```bash
python mapAnim.py -h
```

### Class data file

This file contains the necessary information to link integer values from the 
raster to a description, a hexadecimal color code, and an animation target (the
center of the spiral). It should be formatted as a comma-separated values (CSV)
file and must include a mandatory header with the following fields:

- value
- description
- color
- target_x
- target_y

The order of these fields is flexible, but it's crucial to use the correct 
field names as specified above. You can refer to the example file for the 
proper formatting.


### Elements file

This file contains all the additional elements that can be plotted alongside
the raster map. It adheres to a specific JSON format specifying the element 
type, position, and optional labeling. It also accommodates extra parameters 
that can be passed to the plotting functions of Matplotlib. Currently, the 
script supports the following elements:

- Text
- Lines
- Scalebar
- North Arrow

Please note that the size of the scalebar must be calculated based on the 
raster's resolution. For example, if your original raster has a 90-meter 
resolution, a 250-kilometer scale would require a size of 250,000 / 90 = 
2,778 units. If the raster is resampled, the scale size should be 
recalculated based on the resampled resolution.

For more detailed information, please refer to the example file provided

## Example

The provided example employs data from the [world biomes](https://ecoregions.appspot.com/)
from [Dinerstein et al (2017)](https://academic.oup.com/bioscience/article/67/6/534/3102935).

The original data was converted into a raster format with a 1000-meter 
resolution and projected to a Natural Earth projection. It's worth noting that
this example does not provide latitude-corrected proportions (which might be 
addressed in a future version of the script). However, it effectively 
demonstrates the process of animating a map with the script.

To better illustrate the usage of different fonts, the script uses the free
google font Montserrat (light and semi-bold). You need to get the fonts in your
working directory first. You can obtain the fonts [here](https://fonts.google.com/specimen/Montserrat).

To run the example, install the necessary dependencies, copy the script to 
your working directory, create an output folder named "frames" and execute the
following command:

```bash
python mapAnim.py -r 4 -s 3840 2160 example/biomes.tif example/classdata.csv example/elements.json frames
```


![](assets/biomes.gif)
