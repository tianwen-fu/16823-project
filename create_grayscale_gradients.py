import os

import cv2
import numpy as np

TEMPLATE_IMAGE = (
    os.path.dirname(os.path.abspath(__file__)) + "/data/checkerboard_pattern.png"
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/output/"
LEVELS = 15


def main():
    img = cv2.imread(TEMPLATE_IMAGE, cv2.IMREAD_GRAYSCALE)
    for i, grayval in enumerate(np.linspace(0, 255, LEVELS, dtype=np.uint8)):
        img_new = img.copy()
        img_new[img_new == 0] = grayval
        cv2.putText(
            img=img_new,
            text=f"Black squares in this image are {grayval}/255",
            org=(100, 300),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=5,
            color=0,
            thickness=20,
        )
        cv2.imwrite(f"{OUTPUT_DIR}{i:02d}.png", img_new)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


if __name__ == "__main__":
    main()