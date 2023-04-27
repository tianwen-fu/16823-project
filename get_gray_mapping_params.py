import numpy as np
from matplotlib import pyplot as plt

from camera_utils import (
    DepthData,
    EventLoopHandler,
    capturing_camera,
    convert_lens_params,
)


class Histogram(EventLoopHandler):
    def __init__(self, frames_to_capture: int, frames_per_capture: int = 1):
        super().__init__(1)
        self._captured_frames = []
        self._frames_to_capture = frames_to_capture
        self._frames_per_capture = frames_per_capture
        self._counter = 0
        self._hist_data = None

    @property
    def hist_data(self):
        if self._hist_data is None:
            raise ValueError("No histogram data available. Run the event loop first.")
        return self._hist_data

    def _process_data(self, data: DepthData) -> bool:
        if self._counter % self._frames_per_capture == 0:
            self._captured_frames.append(data.grayscale)
            if len(self._captured_frames) == self._frames_to_capture:
                self._hist_data = np.stack(self._captured_frames, axis=0).flatten()
                return False
        self._counter += 1
        return True


def main():
    handler = Histogram(frames_to_capture=10, frames_per_capture=50)
    with capturing_camera() as (cam, data_queue):
        while handler.event_loop(data_queue):
            pass
        plt.hist(handler.hist_data, bins=100)
        plt.savefig("gray_hist.png")


if __name__ == "__main__":
    main()
