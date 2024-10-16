import npmps
import differential as diff
import polynomial as pn
import numpy as np

# Lz operator in 2D
# N is the number of sites for each dimension
# The first N sites are for x, and the rest sites are for y
def Lz_MPO (N, x1, x2):
    p = diff.diff_MPO (N, x1, x2)
    p = npmps.change_dtype (p, complex)
    p[0] *= -1j

    x = pn.make_x_mpo (N, x1, x2)

    xpy = npmps.direct_product_2MPO (x, p)
    ypx = npmps.direct_product_2MPO (p, x)
    xpy = npmps.change_dtype (xpy, complex)
    ypx = npmps.change_dtype (ypx, complex)
    ypx[0] *= -1

    Lz = npmps.sum_2MPO (xpy, ypx)
    return Lz
