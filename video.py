from pathlib import Path
import os
from cnocr import CnOcr
import json
from pprint import pprint
from tqdm import tqdm

# video_path = Path("public", "videos", "wechat_low_res.mp4")

ocr = CnOcr(det_model_name="db_resnet18")


def scan_one_frame(frame_path: Path):
    out = ocr.ocr(frame_path)
    return [i["text"] for i in out]


if __name__ == "__main__":
    pass
    # os.makedirs("cache", exist_ok=True)
    # os.system(f"ffmpeg -i {video_path} -vf fps=1 cache/%d.png")
    pngs = list(Path("cache").glob("*.png"))
    init = []
    for png in tqdm(pngs):
        init.append(scan_one_frame(png))

    # store in json
    with open("ocr.json", "w") as f:
        json.dump(init, f)
