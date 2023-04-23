import argparse
import queue
import sys
import threading

import cv2
import numpy as np

from roypy import roypy
from roypy.roypy_platform_utils import PlatformHelper
from roypy.roypy_sample_utils import CameraOpener, add_camera_opener_options


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


class DataListener(roypy.IDepthDataListener):
    def __init__(self, q):
        super(DataListener, self).__init__()
        self.frame = 0
        self.done = False
        self.undistortImage = False
        self.lock = threading.Lock()
        self.once = False
        self.queue = q

    def onNewData(self, data):
        p = data.npoints()
        # file:///C:/Users/kevin/projects/royale-lib/doc/html/a00965.html#abfc275fb52656acdce57b5d04e251e2d
        self.queue.put(p)

    def paint(self, data):
        """Called in the main thread, with data containing one of the items that was added to the
        queue in onNewData.
        """
        # mutex to lock out changes to the distortion while drawing
        self.lock.acquire()

        depth = data[:, :, 2]
        gray = data[:, :, 4]
        confidence = data[:, :, 5]

        zImage = np.zeros(depth.shape, np.float32)
        grayImage = np.zeros(depth.shape, np.float32)

        # iterate over matrix, set zImage values to z values of data
        # also set grayImage adjusted gray values
        xVal = 0
        yVal = 0
        for x in zImage:
            for y in x:
                if confidence[xVal][yVal] > 0:
                    zImage[xVal, yVal] = self.adjustZValue(depth[xVal][yVal])
                    grayImage[xVal, yVal] = self.adjustGrayValue(gray[xVal][yVal])
                yVal = yVal + 1
            yVal = 0
            xVal = xVal + 1

        zImage8 = np.uint8(zImage)
        grayImage8 = np.uint8(grayImage)

        # apply undistortion
        # if self.undistortImage:
        #     zImage8 = cv2.undistort(zImage8,self.cameraMatrix,self.distortionCoefficients)
        #     grayImage8 = cv2.undistort(grayImage8,self.cameraMatrix,self.distortionCoefficients)

        # finally show the images
        cv2.imshow("Depth", zImage8)
        cv2.imshow("Gray", grayImage8)

        self.lock.release()
        self.done = True

    def toggleUndistort(self):
        self.lock.acquire()
        self.undistortImage = not self.undistortImage
        self.lock.release()

    # Map the depth values from the camera to 0..255
    def adjustZValue(self, zValue):
        clampedDist = min(2.5, zValue)
        newZValue = clampedDist / 2.5 * 255
        return newZValue

    # Map the gray values from the camera to 0..255
    def adjustGrayValue(self, grayValue):
        clampedVal = min(600, grayValue)
        newGrayValue = clampedVal / 600 * 255
        return newGrayValue

    def setLensParameters(self, lensParameters):
        self.cameraMatrix, self.distortionCoefficients = convert_lens_params(
            lensParameters
        )


def main():
    # Set the available arguments
    platformhelper = PlatformHelper()
    parser = argparse.ArgumentParser(usage=__doc__)
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

    q = queue.Queue()
    l = DataListener(q)
    cam.registerDataListener(l)
    cam.startCapture()

    lensP = cam.getLensParameters()
    l.setLensParameters(lensP)

    process_event_queue(q, l)

    cam.stopCapture()
    print("Done")


def process_event_queue(q, painter):
    while True:
        try:
            # try to retrieve an item from the queue.
            # this will block until an item can be retrieved
            # or the timeout of 1 second is hit
            if len(q.queue) == 0:
                item = q.get(True, 1)
            else:
                for i in range(0, len(q.queue)):
                    item = q.get(True, 1)
        except queue.Empty:
            # this will be thrown when the timeout is hit
            break
        else:
            painter.paint(item)
            # waitKey is required to use imshow, we wait for 1 millisecond
            currentKey = cv2.waitKey(1)
            if currentKey == ord("d"):
                painter.toggleUndistort()
            # close if escape key pressed
            if currentKey == 27:
                break


if __name__ == "__main__":
    main()
