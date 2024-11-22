import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random

def change_path(path):
    frame_path = os.path.join('/scratch/iddp/PCIP',path)
    return frame_path


def plot_bb(frame,bb):
    image = cv2.imread(frame)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create figure and axis
    dpi = 100
    height, width, _ = image.shape
    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
    ax.imshow(image)

    # dot_size = 20
    # line_width = 2

    # Plot initial bounding box
    x1, y1, x2, y2 = bb
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='yellow', linewidth=2)
    ax.add_patch(rect)
    plt.savefig('testone.png')


# plot_bb('/scratch/iddp/PCIP/data/IDDPedestrian/images/gopro/gp_set_0002/gp_set_0002_vid_0001/20160.png',[993.0, 776.57, 1100.21, 1068.9])
plot_bb('/scratch/iddp/PCIP/data/IDDPedestrian/images/gopro/gp_set_0009/gp_set_0009_vid_0001/00356.png',[695.13, 519.14, 739.84, 669.98])
# plot_bb('/scratch/ruthvik/PCIP/data/IDDPedestrian/images/gopro/gp_set_0008/gp_set_0008_vid_0005/03753.png',[ 922.89, 704.57, 991.74, 867.42])

    
