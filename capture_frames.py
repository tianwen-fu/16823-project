import pickle
from argparse import ArgumentParser

import cv2
import numpy as np

from camera_utils import (
    DepthData,
    EventLoopHandler,
    capturing_camera,
    convert_lens_params,
)


class OpenCVPlayback(EventLoopHandler):
    def __init__(self, lens_params, timeout: float):
        super().__init__(timeout)
        self.camera_matrix, self.distortion_coefficients = convert_lens_params(
            lens_params
        )
        self.captured_frames = []

        # region calibrate under each lighting condition, see get_gray_mapping_params.py
        self._gray_scale_low = 50
        self._gray_scale_high = 1000
        # endregion

    @property
    def gray_mapping_params(self):
        return self._gray_scale_low, self._gray_scale_high

    def _gray_map(self, gray_image: np.ndarray):
        low, high = self._gray_scale_low, self._gray_scale_high
        gray_image = gray_image.astype(np.float32).clip(low, high)
        gray_image = (gray_image - low) / (high - low) * 255

        return np.asarray(gray_image, dtype=np.uint8)

    def _process_data(self, data: DepthData) -> bool:
        gray_16u = np.asarray(data.grayscale, dtype=np.uint16)
        gray_16u = cv2.undistort(
            data.grayscale, self.camera_matrix, self.distortion_coefficients
        )
        gray_mapped = self._gray_map(gray_16u)
        # make it a three-channel image for drawing corners
        gray_mapped = cv2.cvtColor(gray_mapped, cv2.COLOR_GRAY2BGR)

        # depth max = 1m; so we times it by 255
        depth_8u = np.asarray(data.coords[..., 2] * 255, dtype=np.uint8)
        depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_WINTER)
        found, corners = cv2.findChessboardCorners(gray_mapped, (14, 9))
        title_appendix = ""
        if found:
            cv2.drawChessboardCorners(gray_mapped, (14, 9), corners, found)
            title_appendix += " (found)"
        # do some upscaling, just to make everything easier to see
        gray_mapped = cv2.resize(
            gray_mapped, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST
        )
        depth_color = cv2.undistort(
            depth_color, self.camera_matrix, self.distortion_coefficients
        )
        cv2.imshow("Gray", gray_mapped)
        cv2.setWindowTitle(
            "Gray",
            f"Gray {gray_16u.shape[1]}x{gray_16u.shape[0]}, min={gray_16u.min()}, max={gray_16u.max()} {title_appendix}",
        )
        cv2.imshow("Depth", depth_color)

        key = cv2.waitKey(20)
        if key == -1:
            return True
        elif chr(key) == "c" or chr(key) == "f":
            # c: capture, f: force capture
            if found or chr(key) == "f":
                print(f"[{len(self.captured_frames)}] collected a frame")
                self.captured_frames.append(data)
            return True
        else:
            cv2.destroyAllWindows()
            return False


def main():
    with capturing_camera() as (cam, data_queue):
        save_path = input("Save path: ")
        handler = OpenCVPlayback(cam.getLensParameters(), 5)
        while handler.event_loop(data_queue):
            pass
        with open(save_path, "wb") as f:
            pickle.dump(
                dict(
                    frames=handler.captured_frames,
                    gray_mapping=handler.gray_mapping_params,
                    camera_matrix=handler.camera_matrix,
                    distortion_coefficients=handler.distortion_coefficients,
                ),
                f,
            )
    print("Done")


if __name__ == "__main__":
    main()
