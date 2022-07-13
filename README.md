# Video-Based Gaze Tracking Tutorial

This is a repository for the Video-Based Gaze Tracking tutorial of the [Bridging the Technological Gap Summer Workshop](https://psychandneuro.duke.edu/opportunity/bridging-technological-gap-summer-workshop), July 31st – August 6th, 2022, German Primate Center, Göttingen, Germany.

Made by Petr Kellnhofer, 2022.

## Preparation

Please take these steps before the start of the tutorial.

**This tutorial is currently under development. Please pull the latest changes before the tutorial (`git pull`).**

### Requirements

The tutorial has been tested on Windows 10 x64 and on GNU/Linux Ubuntu 20.04 but it should work on any computer for which python and required packages are available (e.g. MacOS).
The tutorial has been designed to run on CPU alone, so CUDA (and Nvidia GPU) is not required but it can optionally be used for a better performance.
Some of the tasks utilize a webcam but alternative solutions using provided sample images are also available.

### Environment

The tutorial is implemented as a jupyter notebook and utilizes python 3.9 and pytorch 1.10. It is recommended to create a new python environment using the package manager [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://www.anaconda.com/) and the following steps:

1. Open you terminal with `conda` and navigate to the folder containing this `README.md`.
2. `conda env create -f environment.yml`
3. `conda activate gaze`

Refer to [conda manual]() for more details if needed.

### Data

Download the zip file with data separately from [LINK COMMING SOON](...) and unzip its content *directly* to the folder `data` such that there are folders `data/samples`, etc.

### Help

Use the `Issues` tab on github to ask for help.

