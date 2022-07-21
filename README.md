*See the [Workshop Webpage](https://www.primate-cognition.eu/de/veranstaltungen/bridging-the-technological-gap-workshop.html) for the context of this tutorial.* <br>
*See the [Workshop Hands-on Instructions](https://www.primate-cognition.eu/de/veranstaltungen/bridging-the-technological-gap-workshop/hands-on-sessions) for initial installation steps.*

# Video-Based Gaze Tracking Tutorial

This is a repository for the Video-Based Gaze Tracking tutorial of the [Bridging the Technological Gap Summer Workshop](https://psychandneuro.duke.edu/opportunity/bridging-technological-gap-summer-workshop), July 31st – August 6th, 2022, German Primate Center, Göttingen, Germany.

Made by Petr Kellnhofer, 2022.

## Preparation

Please take these steps before the start of the tutorial. Also please pull the latest code update (`git pull`) before the tutorial starts in case of last minute changes.

### Requirements

The tutorial has been tested on Windows 10 x64 and on GNU/Linux Ubuntu 20.04 but it should work on any computer for which python and required packages are available (e.g. MacOS).
The tutorial has been designed to run on CPU alone, so CUDA (and Nvidia GPU) is not required but it can optionally be used for a better performance.
Some of the tasks utilize a webcam but alternative solutions using provided sample images are also available.

### Environment

The tutorial is implemented as a jupyter notebook and utilizes python 3.9 and pytorch 1.10. It is recommended to create a new python environment using the package manager [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://www.anaconda.com/) and the following steps:

1. Open you terminal with `conda` and navigate to the folder containing this `README.md`.
2. `conda env create -f environment.yml`
3. `conda activate gaze`

Refer to [conda manual](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more details if needed.

Note that the environment is the same as the one used for the [classification tutorial](https://github.com/ccp-eva/tuto_classification) by Pierre-Etienne, so you only need to create it once.


### Data

Download the zip file with data separately from [Google Drive](https://drive.google.com/file/d/1-1VIHfi4s_bxiRvk9-cKdlnl4DaF_YzM/view?usp=sharing) and unzip its content *directly* to the folder `data` such that there are folders `data/samples`, etc.

### Help

Use the `Issues` tab on github to ask for help.

## Usage

1. Open you terminal with `conda` and navigate to the folder containing this `README.md`.
2. `cd notebooks`
3. `conda activate env_workshop`
4. `jupyter notebook`

Follow instructions in [tutorial_instructions.pptx](https://github.com/erkil1452/gaze_workshop_public/blob/master/tutorial_instructions.pptx) for further instructions.

## Google Colab

In case of issues with local environment a slightly modified version of the tutorial is available on [Google Colab](https://colab.research.google.com/drive/193dAJx_sDx1f6rO6_JtVmbyJzomxzYQY?usp=sharing). 

By default, GPU is disabled which prevents the CNN face detector from working. `Runtime`->`Change Runtime type` and select `Hardware accelerator = GPU` to enable GPU support (CUDA). However, note that this introduces a limit on working time for free accounts. All other parts of the tutorial are functional without the GPU support.
