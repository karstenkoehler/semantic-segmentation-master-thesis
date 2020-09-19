import numpy as np
import cv2
import os


# normalized difference vegetation index
def ndvi(r, g, b, nir, result_path=""):
    minus = (nir - r).astype(np.float32)
    plus = (nir + r).astype(np.float32)
    ndvi = np.divide(minus, plus, out=np.zeros_like(minus), where=plus != 0)
    ndvi = np.interp(ndvi, (-1, 1), (0, 255))
    cv2.imwrite(os.path.join(result_path, "ndvi.png"), ndvi)


# ratio vegetation index
def rvi(r, g, b, nir, result_path=""):
    r = r.astype(np.float32)
    nir = nir.astype(np.float32)
    rvi = np.divide(nir, r, out=np.zeros_like(nir), where=r != 0)
    rvi = np.interp(rvi, (0, np.quantile(rvi, 0.99)), (0, 255))
    cv2.imwrite(os.path.join(result_path, "rvi.png"), rvi)


# enhanced vegetation index
def evi(r, g, b, nir, result_path=""):
    x = (nir - r).astype(np.float32)
    y = (nir + 6 * r - 7.5 * b + 1).astype(np.float32)
    evi = np.divide(x, y, out=np.zeros_like(x), where=y != 0) * 2.5
    evi = np.interp(evi, (-1, 1), (0, 255))
    cv2.imwrite(os.path.join(result_path, "evi.png"), evi)


# soil-adjusted vegetation index
def savi(r, g, b, nir, l=0.5, result_path=""):
    savi = ((nir - r) / (nir + r + l)) * (1 + l)
    savi = np.interp(savi, (0, np.quantile(savi, 0.99)), (0, 255))
    cv2.imwrite(os.path.join(result_path, "savi.png"), savi)


# modified soil-adjusted vegetation index
def msavi(r, g, b, nir, result_path=""):
    msavi = (2 * nir + 1 - np.sqrt(np.square(2 * nir + 1) - 8 * nir - r)) / 2
    msavi = np.interp(msavi, (np.min(msavi), np.max(msavi)), (0, 255))
    cv2.imwrite(os.path.join(result_path, "msavi.png"), msavi)


base_dir = os.path.join("E:", "data", "vegetation_analysis")
files = [
    "00_dop10rgbi_32424_5704_1_nw.tif",
    "0D_dop10rgbi_32424_5701_1_nw.tif",
    "1A_dop10rgbi_32425_5698_1_nw.tif",
    "1C_dop10rgbi_32425_5700_1_nw.tif",
    "1E_dop10rgbi_32425_5702_1_nw.tif",
]

for file_name in files:
    file_path = os.path.join(base_dir, file_name)
    result_path = os.path.join(base_dir, file_name[:-4] + "_result")
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    print(file_name)
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    b, g, r, a = cv2.split(image)

    cv2.imwrite(os.path.join(result_path, "original.png"), np.stack([b, g, r], axis=2))
    cv2.imwrite(os.path.join(result_path, "red.png"), r)
    cv2.imwrite(os.path.join(result_path, "green.png"), g)
    cv2.imwrite(os.path.join(result_path, "blue.png"), b)
    cv2.imwrite(os.path.join(result_path, "alpha.png"), a)

    ndvi(r, g, b, a, result_path=result_path)
    rvi(r, g, b, a, result_path=result_path)
    evi(r, g, b, a, result_path=result_path)
    savi(r, g, b, a, result_path=result_path)
    msavi(r, g, b, a, result_path=result_path)
