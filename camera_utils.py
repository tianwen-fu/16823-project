import abc
import argparse
import os
import queue
import sys
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import numpy as np

from roypy import roypy
from roypy.roypy_platform_utils import PlatformHelper
from roypy.roypy_sample_utils import CameraOpener, add_camera_opener_options

if os.name == "nt":
    import pythoncom


@contextmanager
def windows_helper():
    if os.name == "nt":
        pythoncom.CoInitialize()
    yield
    if os.name == "nt":
        pythoncom.CoUninitialize()


def print_camera_info(cam, id=None):
    """Display some details of the camera.

    This method can also be used from other Python scripts, and it works with .rrf recordings in
    addition to working with hardware.
    """
    print("====================================")
    print("        Camera information")
    print("====================================")

    if id:
        print("Id:              " + id)
    print("Type:            " + cam.getCameraName())
    print("Width:           " + str(cam.getMaxSensorWidth()))
    print("Height:          " + str(cam.getMaxSensorHeight()))
    print("Operation modes: " + str(cam.getUseCases().size()))

    listIndent = "    "
    noteIndent = "        "

    useCases = cam.getUseCases()
    for u in range(useCases.size()):
        print(listIndent + useCases[u])
        numStreams = cam.getNumberOfStreams(useCases[u])
        if numStreams > 1:
            print(
                noteIndent + "this operation mode has " + str(numStreams) + " streams"
            )

    try:
        lensparams = cam.getLensParameters()
        print("Lens parameters: " + str(lensparams.size()))
        for u in lensparams:
            print(listIndent + "('" + u + "', " + str(lensparams[u]) + ")")
    except:
        print("Lens parameters not found!")

    camInfo = cam.getCameraInfo()
    print("CameraInfo items: " + str(camInfo.size()))
    for u in range(camInfo.size()):
        print(listIndent + str(camInfo[u]))


def convert_lens_params(lens_params):
    # Construct the camera matrix
    # (fx   0    cx)
    # (0    fy   cy)
    # (0    0    1 )
    cameraMatrix = np.zeros((3, 3), np.float32)
    cameraMatrix[0, 0] = lens_params["fx"]
    cameraMatrix[0, 2] = lens_params["cx"]
    cameraMatrix[1, 1] = lens_params["fy"]
    cameraMatrix[1, 2] = lens_params["cy"]
    cameraMatrix[2, 2] = 1

    # Construct the distortion coefficients
    # k1 k2 p1 p2 k3
    distortionCoefficients = np.zeros((1, 5), np.float32)
    distortionCoefficients[0, 0] = lens_params["k1"]
    distortionCoefficients[0, 1] = lens_params["k2"]
    distortionCoefficients[0, 2] = lens_params["p1"]
    distortionCoefficients[0, 3] = lens_params["p2"]
    distortionCoefficients[0, 4] = lens_params["k3"]

    return cameraMatrix, distortionCoefficients


@dataclass
class DepthData:
    width: int
    height: int
    coords: np.ndarray
    noise: np.ndarray
    grayscale: np.ndarray
    confidence: np.ndarray


class DataListener(roypy.IDepthDataListener):
    def __init__(self, q):
        super().__init__()
        self.queue = q

    def onNewData(self, data):
        points = data.npoints()
        wrapped_data = DepthData(
            width=data.width,
            height=data.height,
            coords=np.stack([points[..., 0], points[..., 1], points[..., 2]], axis=-1),
            noise=points[..., 3],
            grayscale=points[..., 4],
            confidence=points[..., 5],
        )
        self.queue.put(wrapped_data)


class EventLoopHandler(metaclass=abc.ABCMeta):
    def __init__(self, timeout: float):
        self.timeout = timeout

    @abc.abstractmethod
    def _process_data(self, data: DepthData) -> bool:
        pass

    def event_loop(self, q: queue.Queue) -> bool:
        try:
            if len(q.queue) == 0:
                item = q.get(True, 1)
            else:
                for i in range(0, len(q.queue)):
                    item = q.get(True, 1)
            return self._process_data(item)
        except queue.Empty:
            return False


@contextmanager
def capturing_camera():
    with windows_helper():
        parser = argparse.ArgumentParser()
        add_camera_opener_options(parser)
        options = parser.parse_args()
        opener = CameraOpener(options)

        try:
            cam = opener.open_camera()
        except:
            print("could not open Camera Interface")
            sys.exit(1)

        cam.setUseCase("MODE_5_45FPS_500")
        print_camera_info(cam)

        try:
            # retrieve the interface that is available for recordings
            replay = cam.asReplay()
            print("Using a recording")
            print("Framecount : ", replay.frameCount())
            print("File version : ", replay.getFileVersion())
        except SystemError:
            print("Using a live camera")

        data_queue = queue.Queue()
        listener = DataListener(data_queue)
        cam.registerDataListener(listener)
        cam.startCapture()

        yield cam, data_queue

        cam.stopCapture()
