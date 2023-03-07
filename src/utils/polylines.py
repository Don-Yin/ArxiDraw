
from pyaxidraw import axidraw
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw

def normalize_polylines(polylines: list[list[tuple[float, float]]]):
    # normalize to x10, y7
    # upper limit (10, 7)

    x_min = 0
    x_max = max([max([point[0] for point in polyline]) for polyline in polylines])
    y_min = 0
    y_max = max([max([point[1] for point in polyline]) for polyline in polylines])

    canvas_ratio = 10 / 7
    input_radio = (x_max - x_min) / (y_max - y_min)

    if input_radio > canvas_ratio:
        # x is the limiting factor
        y_max = y_min + (x_max - x_min) / canvas_ratio
    else:
        # y is the limiting factor
        x_max = x_min + (y_max - y_min) * canvas_ratio

    x_scale = 10 / (x_max - x_min)
    y_scale = 7 / (y_max - y_min)

    for i in range(len(polylines)):
        for j in range(len(polylines[i])):
            polylines[i][j] = (
                polylines[i][j][0] * x_scale,
                polylines[i][j][1] * y_scale,
            )

    return polylines



def draw_polylines(polylines: list[list[tuple[float, float]]]):
    ad = axidraw.AxiDraw()
    ad.interactive()
    ad.options.model = 1
    ad.options.speed_pendown = 80
    if not ad.connect():  # Open serial port to AxiDraw
        quit()

    for i in tqdm(range(len(polylines))):
        polyline = polylines[i]
        ad.moveto(polyline[0][0], polyline[0][1])
        for j in range(len(polyline) - 1):
            ad.lineto(polyline[j + 1][0], polyline[j + 1][1])

    ad.moveto(0, 0)  # Pen-up move, back to origin.
    ad.disconnect()  # Close serial port to AxiDraw


def animate_preview_polylines(polylines: list[list[tuple[float, float]]]):
    os.makedirs("cache", exist_ok=True)

    for i in tqdm(range(len(polylines))):
        img = Image.new("RGB", (1000, 700), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        for j in range(i + 1):
            polyline = polylines[j]
            for k in range(len(polyline) - 1):
                draw.line(
                    (
                        polyline[k][0],
                        polyline[k][1],
                        polyline[k + 1][0],
                        polyline[k + 1][1],
                    ),
                    fill=(0, 0, 0),
                    width=2,
                )

        img.save(f"cache/{i}.png")