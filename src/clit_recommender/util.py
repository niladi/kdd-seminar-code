from os import listdir
from os.path import isdir, join


def iterate_dirs(path, dir_only=True):
    return filter(
        lambda x: (not dir_only) or isdir(x),
        map(lambda s: join(path, s), listdir(path)),
    )


def flat_map(f, xs):
    return (y for ys in xs for y in f(ys))
