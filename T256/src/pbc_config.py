BOX = {"LX": 1.0, "LY": 1.0, "LZ": 1.0}

def wrap(x, LX, LY, LZ):
    x[..., 0] = x[..., 0] % LX
    x[..., 1] = x[..., 1] % LY
    x[..., 2] = x[..., 2] % LZ
    return x

def min_image(delta, LX, LY, LZ):
    out = delta.clone()
    out[..., 0] = ((out[..., 0] + LX / 2) % LX) - LX / 2
    out[..., 1] = ((out[..., 1] + LY / 2) % LY) - LY / 2
    out[..., 2] = ((out[..., 2] + LZ / 2) % LZ) - LZ / 2
    return out