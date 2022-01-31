import videopipeline as vpl
import os


def add_suffix(path, suffix, new_ext='.mp4'):
    return path.replace('.mp4', f'-{suffix}{new_ext}')


if __name__ == "__main__":

    base_dir = r"C:\Users\johny\myCloud\sandbox\mv_extractor"
    video_path = r"videos/1.mp4"
    clean_path = add_suffix(video_path, "clean")
    mv_path = add_suffix(video_path, "clean", ".csv")
    map_path = add_suffix(video_path, "map")
    docker_cmd = f'docker exec video_cap_dev python3 test.py {clean_path} {mv_path}'

    video_reader = vpl.nodes.generators.ReadVideoFile()
    greyscale = vpl.nodes.functions.Greyscale()(video_reader)
    cropped = vpl.nodes.functions.Crop((400, 900), (1080 - 400, 1920 - 900))(greyscale)
    smoothed = vpl.nodes.functions.Smooth(101)(cropped)
    abs_diff = vpl.nodes.functions.AbsDiff()(smoothed)
    threshold = vpl.nodes.functions.Threshold(32)(abs_diff)
    agg_rgb = vpl.nodes.functions.Greyscale2Rgb(aggregate=True)(threshold)
    writer = vpl.nodes.actions.VideoWriter2(os.path.join(base_dir, clean_path), 30)(agg_rgb)
    system = vpl.nodes.actions.System(docker_cmd)(writer)

    p = vpl.core.Pipeline(system)
    p(os.path.join(base_dir, video_path))

    """
    agg_movement = vpl.nodes.generators.ReadMovementData(aggregate=True)

    flat = vpl.nodes.generators.Flatten(False)([agg_rgb, agg_movement])
    mv_map = vpl.nodes.functions.MovementMap(101)(flat)
    temp_smoothed = vpl.nodes.functions.TemporalSmooth(aggregate=True)(mv_map)
    frames = vpl.nodes.actions.VideoWriter2(os.path.join(base_dir, map_path), 30)(temp_smoothed)

    frames.print_model()
    frames([os.path.join(base_dir, video_path), os.path.join(base_dir, mv_path)])
    """
