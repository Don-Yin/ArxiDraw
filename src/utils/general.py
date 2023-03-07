
from pyaxidraw import axidraw

def reset_to_home():
    ad = axidraw.AxiDraw()
    ad.plot_setup()
    ad.options.mode = "manual"
    ad.options.manual_cmd = "walk_home"
    ad.plot_run()


def manual_reset():
    ad = axidraw.AxiDraw()
    ad.plot_setup()
    ad.options.mode = "align"
    ad.plot_run()