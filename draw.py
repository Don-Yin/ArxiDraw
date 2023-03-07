from pyaxidraw import axidraw
from tqdm import tqdm
import json
import os
from PIL import Image
from PIL import ImageDraw
from pathlib import Path


from src.utils.images import plot_svg
from src.utils.general import manual_reset


if __name__ == "__main__":
    # manual_reset()
    plot_svg("wolf_mask_6300_600.svg")
