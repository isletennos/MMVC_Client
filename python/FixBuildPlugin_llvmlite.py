import numpy as np
from numba.pycc import CC

cc = CC("numba_extensions")

@njit
@cc.export("stuff", "f8(f8[:], f8[:])")
def calc_iou(a1, b2):
    pass


if __name__ == "__main__":
    cc.compile()