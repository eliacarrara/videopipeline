"""

"""
import typing

import cv2
import numpy as np

from videopipeline import core


def crop(frame, position, size):
    assert isinstance(frame, np.ndarray)
    assert isinstance(position, tuple) and len(position) == 2 and all(isinstance(p, int) for p in position)
    assert isinstance(size, tuple) and len(size) == 2 and all(isinstance(s, int) for s in size)

    return frame[position[0]:position[0]+size[0], position[1]:position[1]+size[1]]


def smooth(frame, window):
    assert isinstance(frame, np.ndarray)
    assert isinstance(window, int)

    out_frame = np.zeros_like(frame)
    cv2.GaussianBlur(frame, (window, window), 0, out_frame, 0, cv2.BORDER_CONSTANT)
    return out_frame


def rgb_to_greyscale(frame):
    assert isinstance(frame, np.ndarray)
    return np.array(0.0721 * frame[:, :, 0] + 0.7154 * frame[:, :, 1] + 0.2125 * frame[:, :, 2])  # BRG


def greyscale_to_rgb(frame):
    assert isinstance(frame, np.ndarray)
    return np.dstack([frame, frame, frame])


def filter_largest_contour(contours):
    assert isinstance(contours, tuple), type(contours)
    if len(contours) == 0:
        return None
    else:
        return max(contours, key=lambda c: cv2.contourArea(c))


def get_contour_center(contour):
    if contour is None:
        return tuple()
    else:
        mom = cv2.moments(contour)
        return int(mom["m10"] / mom["m00"]), int(mom["m01"] / mom["m00"])


def draw_contour_centers(frame, center):
    assert isinstance(frame, np.ndarray)
    # TODO argument check

    if center is tuple():
        return frame
    else:
        out_frame = np.array(frame)
        cv2.circle(out_frame, center, 10, (255, 0, 255), -1)

        return out_frame


def draw_line(frame, start_pos, end_pos, color, thickness=3):
    assert isinstance(frame, np.ndarray)
    # TODO argument check

    out_frame = np.array(frame)
    cv2.line(out_frame, (start_pos[0], start_pos[1]), (end_pos[0], end_pos[1]), color, thickness)

    return out_frame


def threshold(frame, t):
    assert isinstance(frame, np.ndarray)
    assert isinstance(t, int)

    out_frame = np.zeros_like(frame)
    cv2.threshold(frame, t, frame.max(initial=0), cv2.THRESH_BINARY, out_frame)

    return out_frame


def erode(frame, kernel):
    assert isinstance(frame, np.ndarray)
    assert isinstance(kernel, np.ndarray)

    out_frame = np.zeros_like(frame)
    cv2.erode(frame, kernel, out_frame)

    return out_frame


def dilate(frame, kernel):
    assert isinstance(frame, np.ndarray)
    assert isinstance(kernel, np.ndarray)

    out_frame = np.zeros_like(frame)
    cv2.dilate(frame, kernel, out_frame)

    return out_frame


def canny_edge(frame, t1, t2):
    assert isinstance(frame, np.ndarray)
    assert isinstance(t1, int)
    assert isinstance(t2, int)

    in_frame = frame.astype(np.uint8)
    out_frame = np.zeros_like(in_frame)
    cv2.Canny(in_frame, t1, t2, out_frame)

    return out_frame


def find_contours(frame):
    assert isinstance(frame, np.ndarray)

    contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def stack(rows, cols, *images):
    assert isinstance(rows, int)
    assert isinstance(cols, int)
    assert all(isinstance(i, np.ndarray) for i in images)
    assert len(images) <= rows * cols, len(images)

    ref = images[0].shape
    out_image = np.zeros((ref[0] * rows, ref[1] * cols, 3))

    # TODO somethings not right here
    for i in range(rows):
        for j in range(cols):
            idx = i * rows + j
            if idx < len(images):
                img = images[idx]
                sub_img = img if img.ndim == 3 else np.dstack([img, img, img])
                out_image[i * ref[0]: (i+1) * ref[0], j * ref[1]: (j+1) * ref[1]] = sub_img

    return out_image


class Crop(core.Function):
    def __init__(self, position: typing.Tuple[int, int], size: typing.Tuple[int, int], **kwargs):
        super().__init__(lambda frame: crop(frame, position, size), **kwargs)


class Smooth(core.Function):
    def __init__(self, window: int, **kwargs):
        super().__init__(lambda frame: smooth(frame, window), **kwargs)


class Rgb2Greyscale(core.Function):
    def __init__(self, **kwargs):
        super().__init__(rgb_to_greyscale, **kwargs)


class Greyscale2Rgb(core.Function):
    def __init__(self, **kwargs):
        super().__init__(greyscale_to_rgb, **kwargs)


class FilterLargestContour(core.Function):
    def __init__(self, **kwargs):
        super().__init__(filter_largest_contour, **kwargs)


class GetContourCenter(core.Function):
    def __init__(self, **kwargs):
        super().__init__(get_contour_center, **kwargs)


class DrawContourCenters(core.Function):
    def __init__(self, **kwargs):
        super().__init__(draw_contour_centers, **kwargs)


class DrawMovementPath(core.Function):
    def __init__(self, window: int = 5, color_coeff: int = 3, **kwargs):
        super().__init__(self.draw_movement_path, **kwargs)
        self.last_center = None
        self.lines = []
        self.window = window
        self.color_coeff = color_coeff

    def draw_movement_path(self, frame, center):
        frame = greyscale_to_rgb(frame)
        if center == tuple():
            self.last_center = None
        else:
            self.last_center = center if self.last_center is None else self.last_center
            self.lines.append((self.last_center, center))
            self.last_center = center

        if len(self.lines) >= 2:

            last_centers = np.array([line[0] for line in self.lines])
            centers = np.array([line[1] for line in self.lines])

            for lc, c in zip(last_centers, centers):
                b = int(min(abs(c[0] - lc[0]) * self.color_coeff, 255))
                g = int(min(abs(c[1] - lc[1]) * self.color_coeff, 255))
                frame = draw_line(frame, lc, c, (b, g, 0))

        return frame


class Threshold(core.Function):
    def __init__(self, t: int, **kwargs):
        super().__init__(lambda frame: threshold(frame, t), **kwargs)


class Erode(core.Function):
    def __init__(self, window: int, **kwargs):
        kernel = np.ones((window, window), 'uint8')
        super().__init__(lambda frame: erode(frame, kernel), **kwargs)


class Dilate(core.Function):
    def __init__(self, window: int, **kwargs):
        kernel = np.ones((window, window), 'uint8')
        super().__init__(lambda frame: dilate(frame, kernel), **kwargs)


class CannyEdge(core.Function):
    def __init__(self, t1: int, t2: int, **kwargs):
        super().__init__(lambda frame: canny_edge(frame, t1, t2), **kwargs)


class FindContours(core.Function):
    def __init__(self, **kwargs):
        super().__init__(lambda frame: find_contours(frame), **kwargs)


class AbsDiff(core.Function):
    def __init__(self, **kwargs):
        super().__init__(self.abs_diff, **kwargs)
        self.last_frame = None

    def abs_diff(self, frame):
        diff = cv2.absdiff(frame, frame) if self.last_frame is None else cv2.absdiff(frame, self.last_frame)
        self.last_frame = frame
        return diff


class Stack(core.Function):
    def __init__(self, rows: int, cols: int, **kwargs):
        super().__init__(lambda *images: stack(rows, cols, *images), **kwargs)


class RollingMean(core.Function):
    def __init__(self, window: int, **kwargs):
        super().__init__(self.rolling_mean, **kwargs)
        self.values = [None] * window
        self.window = window
        self.ptr = 0
        self.filter = np.ones(window)

    def rolling_mean(self, center):
        if center == tuple():
            self.values = [None] * self.window
        else:
            self.values[self.ptr] = center
            self.ptr = (self.ptr + 1) % self.window

        non_none = np.array(list(filter(lambda v: v is not None, self.values)))
        if non_none.shape[0] == 0:
            return tuple()
        else:
            return tuple(non_none.mean(axis=0, dtype=int))
