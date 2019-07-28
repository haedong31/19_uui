import timeit


def my_timer(fn, *args):
    start = timeit.default_timer()
    y = fn(*args)
    end = timeit.default_timer()
    print('[INFO] work time: {} min'.format((end - start) / 60))

    return y
