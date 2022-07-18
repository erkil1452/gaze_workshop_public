
"""
This is a helper class with utilities for the workshop.

Petr Kellnhofer, 2022
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import cv2
import sklearn.decomposition
import imageio

def to_numpy_image(im):
    """
    Converts image to numpy.
    """
    if isinstance(im, torch.Tensor):
        im = im.permute(1, 2, 0).detach().cpu().numpy()
    return im

def im_pca(im):
    """
    Runs PCA to reduce image dimensionality to 3.
    """
    pca = sklearn.decomposition.PCA(n_components=3)
    pca.fit(im.reshape(-1, im.shape[-1]))
    res = pca.transform(im.reshape(-1, im.shape[-1]))
    vmin = res.min(0, keepdims=True)
    vmax = res.max(0, keepdims=True)
    res = (res - vmin) / (vmax - vmin)
    return res.reshape(*im.shape[:-1], -1)

def colormap_image(im, vmin, vmax, cmap = plt.get_cmap('bwr')):
    """
    Color-maps scalar image.
    """
    im = to_numpy_image(im)
    im = np.clip((im - vmin) / (vmax - vmin), 0, 1)
    if len(im.shape) == 3:
        im = im[...,0]
    return cmap(im)

def show_images(imgs, titles=None):
    """
    Shows images in a row.
    """
    fig, axs = plt.subplots(1, len(imgs), figsize=(16, 8))
    axs = np.atleast_1d(axs)
    for i,im in enumerate(imgs):
        im = to_numpy_image(im)
        if len(im.shape) == 3 and im.shape[2] > 4:
            im = im_pca(im)            
        axs[i].imshow(im, cmap='gray')
        axs[i].axis('off')
        if titles is not None:
            axs[i].set_title(titles[i])

def plot_gaze(gaze_lists, labels = [], images = None, figsize=None):
    """
    Plots gaze samples as points or image tiles.
    """
    fig, axs = plt.subplots(1,1,figsize=figsize)    
    axs.cla()

    val_min = np.array([0,0])
    val_max = np.array([1,1])

    for j, gazes in enumerate(gaze_lists):
        if isinstance(gazes, torch.Tensor):
            gazes = gazes.detach().cpu().numpy()
        if len(gazes.shape) == 1:
            gazes = gazes[None]

        if images is not None:
            for gaze, image in zip(gazes, images[j]):
                im_r = 0.1
                extent = (gaze[0] - im_r, gaze[0] + im_r, gaze[1] - im_r, gaze[1] + im_r)
                axs.imshow((image.detach().permute(1,2,0) * 0.5 + 0.5).numpy(), extent=extent)
            val_min = np.minimum(val_min, gazes.min(0) - im_r)
            val_max = np.maximum(val_max, gazes.max(0) + im_r)
        else:
            val_min = np.minimum(val_min, gazes.min(0))
            val_max = np.maximum(val_max, gazes.max(0))
        axs.scatter(gazes[:,0], gazes[:,1])
        
        prefix = f'{labels[j]} ' if j < len(labels) else ''
        for i, gaze in enumerate(gazes):
            axs.text(*(gaze+0.01), f'{prefix}#{i}')
    #fig.canvas.draw()
    #plt.pause(0.1)
    
    axs.set_xlim(val_min[0], val_max[0])
    axs.set_ylim(val_min[1], val_max[1])
    axs.set_xlabel('X')
    axs.set_ylabel('Y')

def draw_rectangles(im, rects):
    """
    Draws red rectangles over an image.
    """
    for i, rect in enumerate(rects):
        cv2.rectangle(im, (rect.left(), rect.bottom()), (rect.right(), rect.top()), color=(255, 0, 0), thickness=4)
        cv2.putText(im, f'#{i}: {rect}', (rect.left(), rect.top() - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 1, lineType = cv2.LINE_AA)
    return im

class WorkshopDataset:
    """
    Sample data for the workshop.
    """
    def __init__(self, base_path: Path):
        super().__init__()
        self.files = sorted([x for x in (base_path / 'select').iterdir() if x.suffix == '.jpg'])
        with (base_path / 'frames.json').open('r') as fid: frames = np.array(json.load(fid))
        with (base_path / 'dotInfo.json').open('r') as fid: dots = json.load(fid)
        self.labels = []
        for fname in self.files:
            index = np.argwhere(frames == fname.name).flatten()[0]
            gaze = np.array([-dots['XCam'][index], dots['YCam'][index]])
            self.labels += [gaze]

        # Transform gaze labels from the GazeCapture space to an approximately [0, 1] space.
        self.labels = np.stack(self.labels, 0)
        self.labels = (self.labels - [[0, -5.5]]) * [[1 / 2, 1 / 4]]
        self.labels = np.clip(self.labels * 0.5 + 0.5, 0, 1) # [-1,1] -> [0,1]
        
    def get_image(self, index: int):
        return imageio.imread(self.files[index])

    def get_gaze(self, index: int):
        return self.labels[index] 
    
    def __len__(self):
        return len(self.files)

class OurDataset(torch.utils.data.Dataset):
    """
    Simple dataset for training.
    """
    def __init__(self):
        self._images = []
        self._labels = []

    def add(self, image: torch.Tensor, label: torch.Tensor):
        """
        Add image-label pair.
        """
        self._images += [image]
        self._labels += [label]

    def __len__(self):
        return len (self._images)

    def __getitem__(self, idx):
        return self._images[idx], self._labels[idx]