# Videopipeline

A fancy wrapper for [opencv](https://opencv.org/) inspired by the [keras model API](https://keras.io/api/models/model/).

## Install
```
pip install videopipeline
```

## Usage
videopipeline can be used to process any datatype. Nonetheless, its primary use is intended for video data.
The examples below show some use cases. Refer to the Python-Notebook in this repository to 
see a more established usage example.

[//]: <> (TODO test examples)

### Linear pipeline
```py
import videopipeline as vpl

video_path = './path/to/video.mp4'
output_path = './path/to/output.mp4'

# Setup pipeline model
raw_video = vpl.generators.ReadVideoFile(video_path)
grey = vpl.functions.Rgb2Greyscale()(raw_video)
crop = vpl.functions.Crop((100, 200), (500, 500))(grey)
smooth1 = vpl.functions.Smooth(101)(crop)
stats = vpl.core.Action(lambda frame: print(frame.mean(), frame.std()))(smooth1)
writer = vpl.actions.VideoWriter(output_path, 30, aggregate=True, collect=False, verbose=True)(stats)

# Invoke pipe
writer()
```

### Tree pipeline
```py
import videopipeline as vpl

video1_path = './path/to/video1.mp4'
video2_path = './path/to/video2.mp4'
output_path = './path/to/output.mp4'

# Setup pipeline model
raw_video1 = vpl.generators.ReadVideoFile(video1_path)
grey1 = vpl.functions.Rgb2Greyscale()(raw_video1)
smooth1 = vpl.functions.Smooth(101)(grey1)

raw_video2 = vpl.generators.ReadVideoFile(video2_path)
grey2 = vpl.functions.Rgb2Greyscale()(raw_video2)
smooth2 = vpl.functions.Smooth(101)(grey2)
diff = vpl.core.Function(lambda *frames: frames[0] - frames[1])([smooth1, smooth2])  # This node has two parents
writer = vpl.actions.VideoWriter(output_path, 30, aggregate=True, collect=False, verbose=True)(diff)

# Invoke pipe
writer()
```

### Graph pipeline
```py
```

### Filter pipeline
```py
```
