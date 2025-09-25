import rasterio
from rasterio.transform import rowcol, xy
from rasterio.enums import Resampling
import numpy as np
from scipy.ndimage import zoom, gaussian_filter
import matplotlib.pyplot as plt
import os

def read_asc_metadata(filepath):
    with rasterio.open(filepath) as src:
        meta = {
            'width': src.width,
            'height': src.height,
            'transform': src.transform,
            'crs': src.crs,
            'nodata': src.nodata
        }
    print(f"Metadata for {filepath}:")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    return meta

def load_elevation(filepath):
    with rasterio.open(filepath) as src:
        arr = src.read(1)
        nodata = src.nodata if src.nodata is not None else -9999
        arr = arr.astype(float)
        arr[arr == nodata] = np.nan
    print(f"Elevation stats: min={np.nanmin(arr):.2f}, max={np.nanmax(arr):.2f}, mean={np.nanmean(arr):.2f}")
    print(f"Nodata count: {np.isnan(arr).sum()}")
    return arr, nodata

def crop_to_bbox(arr, bbox):
    row_start, row_end, col_start, col_end = bbox
    return arr[row_start:row_end, col_start:col_end]

def find_tight_bbox(arr, nodata_val=np.nan, margin=0):
    mask = ~np.isnan(arr) if np.isnan(nodata_val) else (arr != nodata_val)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_idx = np.where(rows)[0]
    col_idx = np.where(cols)[0]
    if len(row_idx) == 0 or len(col_idx) == 0:
        raise ValueError("No valid data found.")
    r0, r1 = row_idx[0], row_idx[-1]+1
    c0, c1 = col_idx[0], col_idx[-1]+1
    # Add margin if needed
    r0 = max(r0 - margin, 0)
    r1 = min(r1 + margin, arr.shape[0])
    c0 = max(c0 - margin, 0)
    c1 = min(c1 + margin, arr.shape[1])
    return (r0, r1, c0, c1)

def downsample(arr, factor):
    if factor == 1:
        return arr
    zoom_factors = (1/factor, 1/factor)
    arr_ds = zoom(arr, zoom_factors, order=1)
    return arr_ds

def save_geotiff(arr, ref_path, out_path):
    with rasterio.open(ref_path) as src:
        meta = src.meta.copy()
        meta.update({
            'dtype': 'float32',
            'count': 1,
            'nodata': np.nan,
            'width': arr.shape[1],
            'height': arr.shape[0]
        })
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(arr.astype(np.float32), 1)
    print(f"Saved GeoTIFF: {out_path}")

def save_npy(arr, out_path):
    np.save(out_path, arr)
    print(f"Saved .npy: {out_path}")

def plot_preview(arr, out_path, cmap='terrain', hillshade=True):
    plt.figure(figsize=(8,6))
    if hillshade:
        from matplotlib.colors import LightSource
        ls = LightSource(azdeg=315, altdeg=45)
        rgb = ls.shade(arr, cmap=plt.get_cmap(cmap), vert_exag=1, blend_mode='overlay')
        plt.imshow(rgb)
    else:
        plt.imshow(arr, cmap=cmap)
        plt.colorbar(label='Elevation (m)')
    plt.title('Elevation Preview')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved preview PNG: {out_path}")

def elevation_at(arr, row, col):
    return arr[row, col]

def elevation_at_coord(arr, x, y, transform):
    row, col = rowcol(transform, x, y)
    return arr[row, col]

def preprocess_asc(
    asc_path,
    crop_bbox=None,
    auto_crop=False,
    margin=0,
    downsample_factor=1,
    smooth_sigma=None,
    out_tif=None,
    out_npy=None,
    out_png=None
):
    meta = read_asc_metadata(asc_path)
    arr, nodata = load_elevation(asc_path)
    transform = None
    with rasterio.open(asc_path) as src:
        transform = src.transform
    if auto_crop:
        bbox = find_tight_bbox(arr, nodata_val=np.nan, margin=margin)
        arr = crop_to_bbox(arr, bbox)
        print(f"Auto-cropped to bbox: {bbox}")
        crop_bbox = bbox
    elif crop_bbox:
        arr = crop_to_bbox(arr, crop_bbox)
        print(f"Cropped to bbox: {crop_bbox}")
    # Update transform for crop
    if crop_bbox:
        row_start, _, col_start, _ = crop_bbox
        transform = rasterio.Affine(transform.a, transform.b, transform.c + col_start * transform.a,
                                    transform.d, transform.e, transform.f + row_start * transform.e)
    if downsample_factor > 1:
        arr = downsample(arr, downsample_factor)
        print(f"Downsampled by factor {downsample_factor}")
        # Update transform for downsampling
        transform = rasterio.Affine(transform.a * downsample_factor, transform.b, transform.c,
                                    transform.d, transform.e * downsample_factor, transform.f)
    if smooth_sigma:
        arr = gaussian_filter(arr, sigma=smooth_sigma)
        print(f"Applied Gaussian smoothing (sigma={smooth_sigma})")
    if out_tif:
        save_geotiff(arr, asc_path, out_tif)
    if out_npy:
        save_npy(arr, out_npy)
    if out_png:
        plot_preview(arr, out_png)
    print(f"Processed array shape: {arr.shape}")
    # Print center pixel real-world coordinates
    center_row, center_col = arr.shape[0] // 2, arr.shape[1] // 2
    cx, cy = xy(transform, center_row, center_col)
    print(f"Center pixel at row,col=({center_row},{center_col}) maps to coordinates: ({cx:.2f}, {cy:.2f})")
    return arr, transform

# Example usage:
if __name__ == "__main__":
    asc_file = "Data/Whole_HK_DTM_5m.asc"  # update path as needed
    
    # Use the actual peak coordinates found earlier: (830875.00, 830125.00)
    peak_x, peak_y = 830875.00, 830125.00
    print(f"Centering crop around peak at coordinates=({peak_x:.2f},{peak_y:.2f})")
    
    # Center crop around the actual peak location
    half_size = 5000  # meters (10km x 10km crop - larger area)
    px_size = 5
    origin_x, origin_y = 799997.5, 848002.5
    min_x = peak_x - half_size
    max_x = peak_x + half_size
    min_y = peak_y - half_size
    max_y = peak_y + half_size
    col_start = int((min_x - origin_x) / px_size)
    col_end = int((max_x - origin_x) / px_size)
    row_start = int((origin_y - max_y) / px_size)
    row_end = int((origin_y - min_y) / px_size)
    crop_bbox = (row_start, row_end, col_start, col_end)
    arr, transform = preprocess_asc(
        asc_file,
        crop_bbox=crop_bbox,
        auto_crop=False,
        downsample_factor=12,  # higher downsample for larger crop
        smooth_sigma=1.5,
        out_tif="Data/processed_dem.tif",
        out_npy="Data/processed_dem.npy",
        out_png="Data/preview.png"
    )
    # Example: get elevation at row/col and at real-world coordinates
    print("Elevation at (100, 100):", elevation_at(arr, 100, 100))
    x, y = 835000, 819000  # example coordinates, update as needed
    print(f"Elevation at coord ({x}, {y}):", elevation_at_coord(arr, x, y, transform))