import cv2
import numpy as np

from videopipeline import core


def crop(frame, position, size):
    return frame[position[0]:position[0]+size[0], position[1]:position[1]+size[1]]


def smooth(frame, window):
    out_frame = np.zeros_like(frame)
    cv2.GaussianBlur(frame, (window, window), 0, out_frame, 0, cv2.BORDER_CONSTANT)
    return out_frame


def rgb_to_greyscale(frame):  # TODO rgb or brg?
    # mono = 0.2125 * r + 0.7154 * g + 0.0721 * b
    return np.array(0.2125 * frame[:, :, 0] + 0.7154 * frame[:, :, 1] + 0.0721 * frame[:, :, 2])


def greyscale_to_rgb(frame):
    return np.dstack([frame, frame, frame])


def filter_largest_contour(contours):
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


def draw_arrows(frame, motion_vectors):
    for mv in motion_vectors:
        start_pt = (mv[3], mv[4])
        end_pt = (mv[5], mv[6])
        cv2.arrowedLine(frame, start_pt, end_pt, (0, 0, 255), 1, cv2.LINE_AA, 0, 0.1)

    return frame


def draw_contour_centers(frame, center):
    if center is None:
        return frame
    else:
        # TODO cv2.circle(frame, center, 10, (255, 0, 255), -1)
        return frame


def draw_line(frame, start_pos, end_pos, color):
    cv2.line(frame, (start_pos[1], start_pos[0]), (end_pos[1], end_pos[0]), color, 3)
    return frame


def threshold(frame, t):
    cv2.threshold(frame, t, frame.max(), cv2.THRESH_BINARY, frame)
    return frame


def erode(frame, window):
    kernel = np.ones((window, window), 'uint8')
    out_frame = np.zeros_like(frame)
    cv2.erode(frame, kernel, out_frame)
    return out_frame


def dilate(frame, window):
    kernel = np.ones((window, window), 'uint8')
    out_frame = np.zeros_like(frame)
    cv2.dilate(frame, kernel, out_frame)
    return out_frame


def canny_edge(frame, t1, t2):
    out_frame = np.zeros(frame.shape, dtype=np.uint8)
    edges = cv2.Canny(frame.astype(np.uint8), t1, t2 * 2, out_frame)
    return np.asarray(edges[:, :])


def find_contours(frame):
    contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return frame, contours


class Crop(core.Function):
    def __init__(self, position, window, **kwargs):
        super().__init__(lambda frame: crop(frame, position, window), **kwargs)


class Smooth(core.Function):
    def __init__(self, window, **kwargs):
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


class DrawArrows(core.Function):
    def __init__(self, **kwargs):
        super().__init__(draw_arrows, **kwargs)


class DrawContourCenters(core.Function):
    def __init__(self, **kwargs):
        super().__init__(draw_contour_centers, **kwargs)


class DrawMovementPath(core.Function):
    def __init__(self, window=5, color_coeff=3, **kwargs):
        super().__init__(self.draw_movement_path, **kwargs)
        self.last_center = None
        self.lines = []
        self.window = window
        self.color_coeff = color_coeff

    def draw_movement_path(self, data):
        frame, center = data
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

            ones_window = np.ones(self.window)
            mean_lc_y = np.convolve(last_centers[:, 0], ones_window, 'valid') / self.window
            mean_lc_x = np.convolve(last_centers[:, 1], ones_window, 'valid') / self.window
            mean_lc = np.vstack([mean_lc_y, mean_lc_x]).T.astype(int)

            mean_c_y = np.convolve(centers[:, 0], ones_window, 'valid') / self.window
            mean_c_x = np.convolve(centers[:, 1], ones_window, 'valid') / self.window
            mean_c = np.vstack([mean_c_y, mean_c_x]).T.astype(int)

            for lc, c in zip(mean_lc, mean_c):
                b = int(min(abs(c[0] - lc[0]) * self.color_coeff, 255))
                g = int(min(abs(c[1] - lc[1]) * self.color_coeff, 255))
                frame = draw_line(frame, lc, c, (b, g, 0))

        return frame, center


class Threshold(core.Function):
    def __init__(self, t, **kwargs):
        super().__init__(lambda frame: threshold(frame, t), **kwargs)


class Erode(core.Function):
    def __init__(self, window):
        super().__init__(lambda frame: erode(frame, window))


class Dilate(core.Function):
    def __init__(self, window):
        super().__init__(lambda frame: dilate(frame, window))


class CannyEdge(core.Function):
    def __init__(self, t1, t2, **kwargs):
        super().__init__(lambda *args: canny_edge(args[0], t1, t2), **kwargs)


class FindContours(core.Function):
    def __init__(self, **kwargs):
        super().__init__(lambda frame: find_contours(frame), **kwargs)


class TemporalSmooth(core.Function):
    def __init__(self, **kwargs):
        super().__init__(self.temp_smooth, **kwargs)
        self.last_frame = None

    def temp_smooth(self, frame):
        diff = frame if self.last_frame is None else np.array([frame, self.last_frame]).mean(axis=0)
        self.last_frame = frame
        return diff


class AbsDiff(core.Function):
    def __init__(self, **kwargs):
        super().__init__(self.abs_diff, **kwargs)
        self.last_frame = None

    def abs_diff(self, frame):
        diff = cv2.absdiff(frame, frame) if self.last_frame is None else cv2.absdiff(frame, self.last_frame)
        self.last_frame = frame
        return diff
