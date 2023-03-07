from pyaxidraw import axidraw
from pathlib import Path


def plot_svg(file: Path):
    ad = axidraw.AxiDraw()  # Create class instance
    ad.plot_setup(file)  # Load file & configure plot context
    ad.options.model = 1
    ad.options.reordering = 2
    ad.options.speed_pendown = 60
    ad.plot_run()  # Plot the file
