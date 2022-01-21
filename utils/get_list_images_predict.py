import os


def get_image_names(basepath):
    def fn(file): return os.path.splitext(file)[1] in [
        '.jpg', '.jpeg'] and os.path.isfile(os.path.join(basepath, file))
    return [image for image in filter(fn, os.listdir(basepath))]
