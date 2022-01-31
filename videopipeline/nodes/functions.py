from .. import core
import cv2
import numpy as np


def crop(frame, position, size):
    return frame[position[0]:position[0]+size[0], position[1]:position[1]+size[1]]


def smooth(frame, window):
    out_frame = np.zeros_like(frame)
    cv2.GaussianBlur(frame, (window, window), 0, out_frame, 0, cv2.BORDER_CONSTANT)
    return out_frame


def greyscale(frame):
    return np.array(0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2])


def greyscale_to_rgb(frame):
    return np.dstack([frame, frame, frame])


def draw_red_arrows(frame, motion_vectors):
    for mv in motion_vectors:
        start_pt = (mv[3], mv[4])
        end_pt = (mv[5], mv[6])
        cv2.arrowedLine(frame, start_pt, end_pt, (0, 0, 255), 1, cv2.LINE_AA, 0, 0.1)

    return frame


def draw_contour_centers(data):
    frame, contours = data
    if len(contours) == 0:
        return frame, tuple()
    
    L = 10
    biggest = max(contours, key=lambda c: cv2.contourArea(c))
    
    M = cv2.moments(biggest)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(frame, (cX, cY), L, (255, 0, 255), 3)

    return frame, (cY, cX)


def draw_movement_path(frame, start_pos, end_pos, color):
    cv2.line(frame, (start_pos[1], start_pos[0]), (end_pos[1], end_pos[0]), color, 3)
    return frame


def smooth_movement_path():
    mean_lc_y = np.convolve(last_centers[:, 0], np.ones(W), 'valid') / W
    mean_lc_x = np.convolve(last_centers[:, 1], np.ones(W), 'valid') / W
    mean_lc = np.vstack([mean_lc_y, mean_lc_x]).T.astype(int)
    
    mean_c_y = np.convolve(centers[:, 0], np.ones(W), 'valid') / W
    mean_c_x = np.convolve(centers[:, 1], np.ones(W), 'valid') / W
    mean_c = np.vstack([mean_c_y, mean_c_x]).T.astype(int)

def frame_threshold(frame, threshold):
    cv2.threshold(frame, threshold, frame.max(), cv2.THRESH_BINARY, frame)
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


def mv_threshold(motion_vectors, threshold):
    out_motion_vectors = []
    for mv in motion_vectors:
        magnitude = (mv[7] ** 2 + mv[8] ** 2) ** 0.5

        if magnitude < threshold:
            continue

        out_motion_vectors.append(mv)

    return np.array(out_motion_vectors)


def movement_map(movement_vectors, shape, window):
    out_frame = np.zeros(shape)

    for mv in movement_vectors:
        magnitude = (mv[7] ** 2 + mv[8] ** 2) ** 0.5
        coords = max(min(mv[6], out_frame.shape[0] - 1), 0), max(min(mv[5], out_frame.shape[1] - 1), 0)
        out_frame[coords] = [abs(mv[7]), abs(mv[8]), magnitude]

    out_frame[:, :, 0] = smooth(out_frame[:, :, 0], window)
    out_frame[:, :, 1] = smooth(out_frame[:, :, 1], window)
    out_frame[:, :, 2] = smooth(out_frame[:, :, 2], window)

    # TODO needed?
    out_frame = out_frame * 255

    return out_frame


def edge_filter(frame, t1, t2):
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


class Greyscale(core.Function):

    def __init__(self, **kwargs):
        super().__init__(lambda frame: greyscale(frame), **kwargs)


class Greyscale2Rgb(core.Function):

    def __init__(self, **kwargs):
        super().__init__(lambda frame: greyscale_to_rgb(frame), **kwargs)
        

class TemporalSmooth(core.Function):

    def __init__(self, **kwargs):
        super().__init__(self.temp_smooth, **kwargs)
        self.last_frame = None

    def temp_smooth(self, frame):
        diff = frame if self.last_frame is None else np.array([frame, self.last_frame]).mean(axis=0)
        self.last_frame = frame
        return diff
        
        
class Threshold(core.Function):

    def __init__(self, threshold, **kwargs):
        super().__init__(lambda frame: frame_threshold(frame, threshold), **kwargs)


class Erode(core.Function):

    def __init__(self, window):
        super().__init__(lambda frame: erode(frame, window))


class Dilate(core.Function):

    def __init__(self, window):
        super().__init__(lambda frame: dilate(frame, window))


class AbsDiff(core.Function):

    def __init__(self, **kwargs):
        super().__init__(self.abs_diff, **kwargs)
        self.last_frame = None

    def abs_diff(self, frame):
        diff = cv2.absdiff(frame, frame) if self.last_frame is None else cv2.absdiff(frame, self.last_frame)
        self.last_frame = frame
        return diff


class MovementVectorArrowWriter(core.Function):

    def __init__(self, **kwargs):
        super().__init__(draw_red_arrows, **kwargs)


class DrawContourCenters(core.Function):

    def __init__(self, **kwargs):
        super().__init__(draw_contour_centers, **kwargs)


class DrawMovementPath(core.Function):
    
    def __init__(self, **kwargs):
        super().__init__(self.draw_movement_path, **kwargs)
        self.last_center = None
        self.lines = []
        
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
            C = 3
            W = 5
            
            last_centers = np.array([l[0] for l in self.lines])
            centers = np.array([l[1] for l in self.lines])
            
            mean_lc_y = np.convolve(last_centers[:, 0], np.ones(W), 'valid') / W
            mean_lc_x = np.convolve(last_centers[:, 1], np.ones(W), 'valid') / W
            mean_lc = np.vstack([mean_lc_y, mean_lc_x]).T.astype(int)
            
            mean_c_y = np.convolve(centers[:, 0], np.ones(W), 'valid') / W
            mean_c_x = np.convolve(centers[:, 1], np.ones(W), 'valid') / W
            mean_c = np.vstack([mean_c_y, mean_c_x]).T.astype(int)
            
            for lc, c in zip(mean_lc, mean_c):
                b = int(min(abs(c[0] - lc[0]) * C, 255))
                g = int(min(abs(c[1] - lc[1]) * C, 255))
                frame = draw_movement_path(frame, lc, c, (b, g, 0))
        
        return frame, center


class SmoothMovementPath(core.Function):

    def __init__(self, **kwargs):
        super().__init__(self.smooth_movement_path, **kwargs)
   
        
class MotionVectorThreshold(core.Function):

    def __init__(self, **kwargs):
        super().__init__(mv_threshold, **kwargs)


class MovementMap(core.Function):

    def __init__(self, shape, window, **kwargs):
        super().__init__(lambda *args: movement_map(args[0], shape, window), **kwargs)


class EdgeFilter(core.Function):

    def __init__(self, t1, t2, **kwargs):
        super().__init__(lambda *args: edge_filter(args[0], t1, t2), **kwargs)


class FindContours(core.Function):

    def __init__(self, **kwargs):
        super().__init__(lambda *args: find_contours(args[0]), **kwargs)
