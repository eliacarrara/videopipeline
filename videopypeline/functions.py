"""

"""

import typing

import cv2
import numpy as np

from videopypeline import core


def _check_tuple(obj, true_len=2, true_type=int | np.int32):
    assert isinstance(obj, tuple), type(obj)
    assert len(obj) == true_len, len(obj)
    assert all(isinstance(o, true_type) for o in obj), [type(o) for o in obj]
    return True


def _check_nparray(array, ndim=None, dtype=None):
    assert isinstance(array, np.ndarray), type(array)
    assert ndim is None or array.ndim == ndim, array.ndim
    assert dtype is None or array.dtype == dtype, array.dtype
    return True


def crop(frame, position, size):
    assert _check_nparray(frame)
    assert _check_tuple(position)
    assert _check_tuple(size)
    return frame[position[0]:position[0]+size[0], position[1]:position[1]+size[1]]


def affine_transformation(frame, pos_d=None, rot_d=None, scale_d=None):
    assert True if pos_d is None else _check_tuple(pos_d, true_type=float | np.float)
    assert True if rot_d is None else isinstance(rot_d, float)
    assert True if scale_d is None else isinstance(scale_d, float)

    scale_d = 1 if scale_d is None else scale_d
    img_shape = frame.shape[1::-1]
    pivot_pos = np.array(img_shape) / 2 + pos_d

    rot_mat = cv2.getRotationMatrix2D(pivot_pos, rot_d, scale_d)
    result = cv2.warpAffine(frame, rot_mat, img_shape, flags=cv2.INTER_LINEAR)

    return result


def perspective_transformation(frame, src_points, dst_points, out_shape):
    assert _check_nparray(frame)
    assert isinstance(src_points, np.ndarray)
    assert isinstance(dst_points, np.ndarray)
    assert src_points.ndim == 2
    assert dst_points.ndim == 2
    assert _check_tuple(out_shape)

    mat = cv2.getPerspectiveTransform(src_points, dst_points)
    out_frame = cv2.warpPerspective(frame, mat, out_shape)
    return out_frame


def smooth(frame, window_size):
    assert _check_nparray(frame)
    assert isinstance(window_size, int)

    out_frame = np.zeros_like(frame)
    cv2.GaussianBlur(frame, (window_size, window_size), 0, out_frame, 0, cv2.BORDER_CONSTANT)

    return out_frame


def rgb_to_greyscale(frame):
    assert _check_nparray(frame, ndim=3)
    return np.array(0.0721 * frame[:, :, 0] + 0.7154 * frame[:, :, 1] + 0.2125 * frame[:, :, 2])  # BRG


def greyscale_to_rgb(frame):
    assert _check_nparray(frame, ndim=2)
    return np.dstack([frame, frame, frame])


def filter_largest_contour(contours):
    assert isinstance(contours, tuple)

    if len(contours) == 0:
        return None
    else:
        return max(contours, key=cv2.contourArea)


def get_contour_center(contour):
    if contour is None:
        return None
    else:
        assert _check_nparray(contour)
        mom = cv2.moments(contour)
        x, y = np.int32(mom["m10"] / mom["m00"]), np.int32(mom["m01"] / mom["m00"])
        return np.array([x, y])


def find_contours(frame):
    assert _check_nparray(frame, ndim=2)
    contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def convex_contour(contour):
    if contour is None:
        return None
    else:
        assert _check_nparray(contour)
        hull = cv2.convexHull(contour)
        return hull


def bounding_box(contour):
    if contour is None:
        return None
    else:
        return cv2.boundingRect(contour)


def draw_contour_centers(frame, center):
    assert _check_nparray(frame)

    if center is None:
        return frame
    else:
        assert _check_nparray(center)

        out_frame = np.array(frame)
        cv2.circle(out_frame, tuple(center), 10, (255, 0, 255), -1)

        return out_frame


def draw_text(frame, text, org=(100, 100), scale=3, color=(255, 0, 255), thickness=3):
    assert _check_nparray(frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    return cv2.putText(np.array(frame), f'{text}', org, font, scale, color, thickness, cv2.LINE_AA)


def draw_line(frame, start_pos, end_pos, color, thickness=3):
    assert _check_nparray(frame, ndim=3)
    assert _check_tuple(start_pos)
    assert _check_tuple(end_pos)
    assert _check_tuple(color, true_len=3)
    assert isinstance(thickness, int)

    out_frame = np.array(frame)
    cv2.line(out_frame, (start_pos[0], start_pos[1]), (end_pos[0], end_pos[1]), color, thickness)

    return out_frame


def threshold(frame, t):
    assert _check_nparray(frame)
    assert isinstance(t, int)

    out_frame = np.zeros_like(frame)
    cv2.threshold(frame, t, frame.max(initial=0), cv2.THRESH_BINARY, out_frame)

    return out_frame


def erode(frame, kernel):
    assert _check_nparray(frame)
    assert _check_nparray(kernel, ndim=2)

    out_frame = np.zeros_like(frame)
    cv2.erode(frame, kernel, out_frame)

    return out_frame


def dilate(frame, kernel):
    assert _check_nparray(frame)
    assert _check_nparray(kernel, ndim=2)

    out_frame = np.zeros_like(frame)
    cv2.dilate(frame, kernel, out_frame)

    return out_frame


def canny_edge(frame, t1, t2):
    assert _check_nparray(frame, ndim=2)
    assert isinstance(t1, int)
    assert isinstance(t2, int)

    in_frame = frame.astype(np.uint8)
    out_frame = np.zeros_like(in_frame)
    cv2.Canny(in_frame, t1, t2, out_frame)

    return out_frame


def stack(rows, cols, *images):
    assert isinstance(rows, int)
    assert isinstance(cols, int)
    assert all(_check_nparray(i) for i in images)
    assert len(images) <= rows * cols, len(images)

    ref = images[0].shape
    out_image = np.zeros((ref[0] * rows, ref[1] * cols, 3))

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(images):
                img = images[idx]
                sub_img = img if img.ndim == 3 else np.dstack([img, img, img])
                out_image[i * ref[0]: (i+1) * ref[0], j * ref[1]: (j+1) * ref[1]] = sub_img

    return out_image


class Crop(core.Function):
    def __init__(self, position: typing.Tuple[int, int], size: typing.Tuple[int, int], **kwargs):
        super().__init__(lambda frame: crop(frame, position, size), **kwargs)


class AffineTransformation(core.Function):
    def __init__(self, pos_d=None, rot_d=None, scale_d=None, **kwargs):
        super().__init__(lambda frame: affine_transformation(frame, pos_d, rot_d, scale_d), **kwargs)


class PerspectiveTransformation(core.Function):
    def __init__(self, src_points, dst_points, out_shape, **kwargs):
        super().__init__(lambda frame: perspective_transformation(frame, src_points, dst_points, out_shape), **kwargs)


class Smooth(core.Function):
    def __init__(self, window_size: int, **kwargs):
        super().__init__(lambda frame: smooth(frame, window_size), **kwargs)


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


class FindContours(core.Function):
    def __init__(self, **kwargs):
        super().__init__(lambda frame: find_contours(frame), **kwargs)


class ConvexHull(core.Function):
    def __init__(self, **kwargs):
        super().__init__(convex_contour, **kwargs)


class BoundingBox(core.Function):
    def __init__(self, **kwargs):
        super().__init__(bounding_box, **kwargs)


class DrawContourCenters(core.Function):
    def __init__(self, **kwargs):
        super().__init__(draw_contour_centers, **kwargs)


class DrawText(core.Function):
    def __init__(self, org=(100, 100), scale=3, color=(255, 0, 255), thickness=3, **kwargs):
        super().__init__(lambda frame, text: draw_text(frame, f'{text}', org=org, scale=scale, color=color, thickness=thickness), **kwargs)


class DrawMovementPath(core.Function):
    def __init__(self, color_coeff: int = 4, **kwargs):
        super().__init__(self.draw_movement_path, **kwargs)
        self.last_center = None
        self.lines = []
        self.color_coeff = color_coeff

    def draw_movement_path(self, frame, center):
        assert _check_nparray(frame)

        if frame.ndim == 2:
            frame = greyscale_to_rgb(frame)

        if center is not None and self.last_center is None:
            self.lines = []
        elif center is not None and self.last_center is not None:
            assert _check_nparray(center)
            self.lines.append((self.last_center, center))

        if len(self.lines) >= 2:
            lines = np.array(self.lines)
            last_centers, centers = lines[:, 0], lines[:, 1]

            for lc, c in zip(last_centers, centers):
                b = int(min(abs(c[0] - lc[0]) * self.color_coeff, 255))
                g = int(min(abs(c[1] - lc[1]) * self.color_coeff, 255))
                frame = draw_line(frame, tuple(lc), tuple(c), (b, g, 0))

        self.last_center = center
        return frame


class Threshold(core.Function):
    def __init__(self, t: int, **kwargs):
        super().__init__(lambda frame: threshold(frame, t), **kwargs)


class Erode(core.Function):
    def __init__(self, window_size: int, **kwargs):
        kernel = np.ones((window_size, window_size), 'uint8')
        super().__init__(lambda frame: erode(frame, kernel), **kwargs)


class Dilate(core.Function):
    def __init__(self, window_size: int, **kwargs):
        kernel = np.ones((window_size, window_size), 'uint8')
        super().__init__(lambda frame: dilate(frame, kernel), **kwargs)


class CannyEdge(core.Function):
    def __init__(self, t1: int, t2: int, **kwargs):
        super().__init__(lambda frame: canny_edge(frame, t1, t2), **kwargs)


class AbsDiff(core.Function):
    def __init__(self, **kwargs):
        super().__init__(self.abs_diff, **kwargs)
        self.last_frame = None

    def abs_diff(self, frame):
        assert _check_nparray(frame)

        diff = cv2.absdiff(frame, frame) if self.last_frame is None else cv2.absdiff(frame, self.last_frame)
        self.last_frame = frame

        return diff


class Stack(core.Function):
    def __init__(self, rows: int, cols: int, **kwargs):
        super().__init__(lambda *images: stack(rows, cols, *images), **kwargs)
