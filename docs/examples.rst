########
Examples
########

videopypeline can be used to process any datatype. Nonetheless, its primary use is intended for videos.
The examples below show some use cases. Refer to the Python-Notebook in this repository to
see a more established usage example.

..
   TODO test examples


***************
Linear pipeline
***************

.. code-block:: python

    import videopypeline as vpl

    video_path = './path/to/video.mp4'
    output_path = './path/to/output.mp4'

    # Setup pipeline model
    raw_video = videopypeline.generators.ReadVideoFile(video_path)
    grey = videopypeline.functions.Rgb2Greyscale()(raw_video)
    crop = videopypeline.functions.Crop((100, 200), (500, 500))(grey)
    smooth1 = videopypeline.functions.Smooth(101)(crop)
    stats = videopypeline.core.Action(lambda frame: print(frame.mean(), frame.std()))(smooth1)
    writer = videopypeline.actions.VideoWriter(output_path, 30, aggregate=True, collect=False, verbose=True)(stats)

    # Invoke pipeline
    _ = writer()


*************
Tree pipeline
*************

.. code-block:: python

    import videopypeline as vpl

    video1_path = './path/to/video1.mp4'
    video2_path = './path/to/video2.mp4'
    output_path = './path/to/output.mp4'

    # Setup pipeline model
    raw_video1 = videopypeline.generators.ReadVideoFile(video1_path)
    grey1 = videopypeline.functions.Rgb2Greyscale()(raw_video1)
    smooth1 = videopypeline.functions.Smooth(101)(grey1)

    raw_video2 = videopypeline.generators.ReadVideoFile(video2_path)
    grey2 = videopypeline.functions.Rgb2Greyscale()(raw_video2)
    smooth2 = videopypeline.functions.Smooth(101)(grey2)

    # This node has two parents
    diff = videopypeline.core.Function(lambda *frames: frames[0] - frames[1])([smooth1, smooth2])
    writer = videopypeline.actions.VideoWriter(output_path, 30, aggregate=True, collect=False, verbose=True)(diff)

    # Invoke pipeline
    _ = writer()

**************
Graph pipeline
**************

.. code-block:: python

    import videopypeline as vpl

***************
Filter pipeline
***************

.. code-block:: python

    import videopypeline as vpl