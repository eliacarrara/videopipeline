import cv2
import numpy as np

from videopypeline import core


class VideoWriter(core.Action):

    def __init__(self, output_path, fps, **kwargs):
        super().__init__(self.write, **kwargs)
        self.output_path = output_path
        self.fps = fps
        self.writer = None

    def end_callback(self):
        if self.writer is not None:
            self.writer.release()

    def write(self, frame):
        frame = frame[0] if isinstance(frame, tuple) else frame
        assert isinstance(frame, np.ndarray), type(frame)
        if frame.ndim == 2:
            frame = np.dstack([frame, frame, frame])
        elif frame.ndim == 3:
            assert frame.shape[2] == 3
        else:
            assert False

        if self.writer is None:
            shape = tuple(reversed(frame.shape[:2]))
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, shape)

        self.writer.write(frame.astype(np.uint8))


class VideoWriter2(core.Action):

    def __init__(self, output_path, fps):
        super().__init__(self.write)
        self.output_path = output_path
        self.fps = fps

    def write(self, frames):
        frames = frames[0] if isinstance(frames, tuple) else frames
        assert isinstance(frames, list)
        assert all(type(frame) == np.ndarray for frame in frames)
        if len(frames) == 0:
            return

        shape = list(reversed(frames[0].shape[:2]))
        writer = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), self.fps, shape)

        for frame in frames:
            writer.write(frame.astype(np.uint8))

        writer.release()
