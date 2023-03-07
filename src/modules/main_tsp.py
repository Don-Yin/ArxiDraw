import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import cv2
import random
from tqdm import tqdm


class TargetImage:
    def __init__(self, path: Path):
        self.img = Image.open(path).convert("L")
        # thubnail image to fit in 768x768
        self.img.thumbnail((768, 768))
        self.img = np.array(self.img)
        self.filter_to_locs()

    def filter_to_locs(self):
        # count the total number of pixels that is not white
        total_pixels = np.count_nonzero(self.img)

        # get the coordinates of all the pixels
        coords = np.indices(self.img.shape).reshape(2, -1)
        values = self.img.flatten()

        # sort the coordinates by the pixel value
        sorted_coords = coords[:, values.argsort()]

        # get the top 10% darkest pixels
        top_coords = sorted_coords[:, : int(total_pixels * 0.2)]
        top_coords = top_coords.T

        cycle = np.arange(len(top_coords))

        # preview the top 10% of the coordinates
        img_preview = np.zeros(self.img.shape, dtype=np.uint8)
        img_preview.fill(255)
        for coord in top_coords:
            img_preview[coord[0], coord[1]] = 0
        cv2.imwrite("preview.png", img_preview)

        return top_coords, cycle

    def check_line_space(self, coord_1, coord_2):
        # check the pixel values on the line between coord_1 and coord_2
        mask = np.zeros(self.img.shape, dtype=np.uint8)
        cv2.line(mask, tuple(coord_1), tuple(coord_2), (255, 255, 255), thickness=2)
        pixels = self.img[mask == 255]
        # check if more than 90% of the pixels are white
        try:
            mostly_white = np.count_nonzero(pixels > 235) / len(pixels) > 0.06
        except ZeroDivisionError:
            mostly_white = False
        # check if the line is longer than 20 pixels
        long_enough = np.linalg.norm(coord_1 - coord_2) > 24
        return mostly_white and long_enough

    def compare(self, img: np.ndarray) -> float:
        return ssim(self.img, img)

    def make_canvas(self) -> np.ndarray:
        canvas = np.zeros(self.img.shape, dtype=np.uint8)
        canvas.fill(255)
        return canvas


if __name__ == "__main__":
    from modules.tsp import GeneticAlgorithm

    target = TargetImage(Path("leo.png"))

    locs, cycle = target.filter_to_locs()

    num_locs_per_cluster = 2**5

    process_ = GeneticAlgorithm(len(locs), total_num_iterations=3000000)
    new_cycle, length = process_.main(locs, cycle, num_locs_per_cluster)

    # sort locs by cycle
    top_coords = locs[new_cycle]

    img_preview = np.zeros(target.img.shape, dtype=np.uint8)
    img_preview.fill(255)

    # save the coordinates to a file
    np.savez_compressed("leo_coords.npz", top_coords)

    # ----------------------------
    coords = np.load("leo_coords.npz")["arr_0"]
    polylines = []
    polyline = []
    for i in tqdm(range(len(coords) - 1)):
        if target.check_line_space(coords[i], coords[i + 1]):
            if polyline:
                polylines.append(polyline)
                polyline = []
        else:
            # print(coords[i].tolist())
            coord = coords[i].tolist()
            coord.reverse()
            polyline.append(coord)

    # draw the polyline
    canvas = target.make_canvas()
    for polyline in polylines:
        polyline = np.array(polyline)
        cv2.polylines(canvas, [polyline], isClosed=False, color=(0, 0, 0), thickness=1)
    Image.fromarray(canvas).show()

    import json

    # save to json file
    with open("leo.json", "w") as f:
        json.dump(polylines, f)
