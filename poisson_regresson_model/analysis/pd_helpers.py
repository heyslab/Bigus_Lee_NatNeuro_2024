import pandas as pd
import itertools as it
from functools import partial
import numpy as np

def set_index(ser, level):
    return ser.to_frame().set_index(
        ser.index.get_level_values(level))[0 if ser.name is None else ser.name]

def drop_columns(df, mask):
    return df.drop(columns=df.loc[:, mask].columns)

def drop_null(df, level, axis=1, invert=False, fill=None):
    if axis == 0:
        index = df.index
    else:
        index = df.columns

    if type(level) == str:
        _level = [level]
    else:
        _level = list(level)

    if invert:
        mask = index.get_level_values(_level.pop()).isna()
    else:
        mask = index.get_level_values(_level.pop()).notnull()

    while len(_level):
        if invert:
            mask = mask & index.get_level_values(_level.pop()).isna()
        else:
            mask = mask & index.get_level_values(_level.pop()).notnull()

    if fill is not None:
        return df.mask(np.tile(np.logical_not(mask), (df.shape[0], 1)),
                       other=fill)

    return df.loc[(*[slice(None)]*axis, mask)]


def add_level(index, new_level, name, level=-1):
    if level < 0:
        level = len(index.names) + 1 + level
    names = index.names
    if len(names) == 1:
        index = [[a] for a in index.values]

    return pd.MultiIndex.from_tuples(
        map(lambda x: (*x[0][:x[2]], x[1], *x[0][x[2]:]),
        zip(index, new_level, it.repeat(level))),
        names=names[:level] + [name] + names[level:])


def update_level(index, new_vals, name):
    level = index.names.index(name)
    return add_level(index.droplevel(name), new_vals, name, level=level)


def pad_index(index, n):
    return index.append(pd.MultiIndex.from_frame(
        pd.DataFrame(None, index=np.arange(n), columns=index.names)))


def indexer(index, name):
    order = index.names.index(name)
    if order == 0:
        return lambda x: x

    func = lambda y, x: pd.IndexSlice.__getitem__((*[slice(None)]*y, x))
    return partial(func, order)


def index_series(index, name):
    return pd.Series(index.get_level_values(name), index=index.droplevel(name))


def sort(df, order):
    reorder_func = partial(df.T.xs, drop_level=False)
    segs = []
    for idx in order:
        seg = reorder_func(idx)
        if seg.ndim == 1:
            seg = pd.DataFrame(seg).T
            seg.index.names=df.columns.names
        segs.append(seg)
    return pd.concat(segs).T

def tile_mask(df, mask, axis=0, other=np.nan):
    return df.mask(np.tile(mask, [df.shape[axis], 1][::(-1 + 2 * (axis == 0))]),
                     other=other)

def select(df, items, level):
    return df.drop(df.index.unique(level).difference(items), level=level)
