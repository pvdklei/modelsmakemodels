def _conv_out_shape(og_size, filter_size, stride, padding):
    return int((og_size - filter_size + 2 * padding) / stride) + 1


def conv_out_shape(og_size, filter_size, stride=1, padding=0):
    if type(og_size) == tuple:
        og_width = og_size[0]
        og_height = og_size[1]
        return _conv_out_shape(og_width, filter_size, stride, padding), _conv_out_shape(og_height, filter_size, stride, padding)
    else:
        out = _conv_out_shape(og_size, filter_size, stride, padding)
        return out, out


def _convt_out_shape(og_size, filter_size, stride, padding):
    return int((og_size - 1) * stride - 2 * padding + filter_size)


def convt_out_shape(og_size, filter_size, stride=1, padding=0):
    if type(og_size) == tuple:
        og_width = og_size[0]
        og_height = og_size[1]
        return _convt_out_shape(og_width, filter_size, stride, padding), _conv_out_shape(og_height, filter_size, stride, padding)
    else:
        out = _convt_out_shape(og_size, filter_size, stride, padding)
        return out, out


def flatten_size(x, chans):
    return chans * x[0] * x[1]


def intlerp(a, one, other):
    return int(one + a * (other - one))
