import numpy as np
import cv2

file_path = "C:/thesis-data/06_dop10rgbi_32440_5694_1_nw.tif"
image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)


# normalized difference vegetation index
def ndvi(r, g, b, nir):
    minus = (nir - r).astype(np.float32)
    plus = (nir + r).astype(np.float32)
    ndvi = np.divide(minus, plus, out=np.zeros_like(minus), where=plus != 0)
    ndvi = np.interp(ndvi, (-1, 1), (0, 255))
    cv2.imwrite(f"C:/thesis-data/tiles/ndvi.png", ndvi)


# ratio vegetation index
def rvi(r, g, b, nir):
    r = r.astype(np.float32)
    nir = nir.astype(np.float32)
    rvi = np.divide(nir, r, out=np.zeros_like(nir), where=r != 0)
    rvi = np.interp(rvi, (0, np.quantile(rvi, 0.99)), (0, 255))
    cv2.imwrite(f"C:/thesis-data/tiles/rvi.png", rvi)


# enhanced vegetation index
def evi(r, g, b, nir):
    x = (nir - r).astype(np.float32)
    y = (nir + 6 * r - 7.5 * b + 1).astype(np.float32)
    evi = np.divide(x, y, out=np.zeros_like(x), where=y != 0) * 2.5
    evi = np.interp(evi, (-1, 1), (0, 255))
    cv2.imwrite(f"C:/thesis-data/tiles/evi.png", evi)


# soil-adjusted vegetation index
def savi(r, g, b, nir, l=0.5):
    savi = ((nir - r) / (nir + r + l)) * (1 + l)
    savi = np.interp(savi, (0, np.quantile(savi, 0.99)), (0, 255))
    cv2.imwrite(f"C:/thesis-data/tiles/savi.png", savi)


# modified soil-adjusted vegetation index
def msavi(r, g, b, nir):
    msavi = (2 * nir + 1 - np.sqrt(np.square(2 * nir + 1) - 8 * nir - r)) / 2
    msavi = np.interp(msavi, (np.min(msavi), np.max(msavi)), (0, 255))
    cv2.imwrite(f"C:/thesis-data/tiles/msavi.png", msavi)


r, g, b, a = cv2.split(image)

cv2.imwrite(f"C:/thesis-data/tiles/red.png", r)
cv2.imwrite(f"C:/thesis-data/tiles/green.png", g)
cv2.imwrite(f"C:/thesis-data/tiles/blue.png", b)
cv2.imwrite(f"C:/thesis-data/tiles/alpha.png", a)

ndvi(r, g, b, a)
rvi(r, g, b, a)
evi(r, g, b, a)
savi(r, g, b, a)
msavi(r, g, b, a)
