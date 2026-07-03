# GeoTIFF Single-File Inspection Notes

## Purpose

The goal of this inspection was to understand what is contained inside a CRASAR-U-DROIDs `.geo.tif` file and how it differs from ordinary video/frame data.

## Selected file

The selected file was:

```text
train/imagery/CREWED/090401-DMS-Assessment-Westpark.geo.tif_20170830_RGB.geo.tif

This file was selected because it is hurricane-related, RGB, georeferenced, and small enough to download locally for inspection.

Why this file is useful

This file is useful for a first GeoTIFF inspection because:

It is related to Hurricane Harvey.
It is CREWED aerial imagery.
It is RGB imagery rather than DSM elevation data.
It is small enough to download and process locally.
It has a corresponding building damage assessment JSON annotation file.
What GeoTIFF inspection demonstrates

A GeoTIFF is not just a normal image. It contains raster pixel values plus geospatial metadata such as:

Width and height.
Number of bands.
Coordinate reference system, or CRS.
Pixel-to-map affine transform.
Geographic/spatial bounds.
Spatial resolution.
Data types and nodata values.

This makes GeoTIFF data different from ordinary video frames. A video frame is usually just an RGB image in a temporal sequence. A GeoTIFF is an image with spatial reference.

Converting GeoTIFF to ordinary image data

The GeoTIFF was also exported to a normal PNG preview. This demonstrates that if we want to use only non-georeferenced image/frame-like data, we can convert:

GeoTIFF raster -> RGB PNG/JPG image

However, if we discard georeferencing for model input, we should preserve the original metadata separately, including:

building_id
label
source
original_geotiff_filename
crop coordinates
CRS
transform
Annotation connection

The corresponding building damage assessment JSON file can be used to connect the imagery to building-level labels and polygons.

The key annotation fields are:

building_id
view_id
label
pixels
boundary
source
filename

This suggests that the correct next step is not only to export full PNG previews, but to extract building-level crops from the GeoTIFF using the annotation pixel polygons.

Next step

The next technical step is to create a script that takes:

GeoTIFF image + building damage assessment JSON

and outputs:

building crop PNG files + metadata CSV

The metadata CSV should include:

crop_path
building_id
view_id
label
source
filename
original_geotiff_path
pixel_bounds

## Actual inspection result

The inspected GeoTIFF had the following properties:

| Field | Value |
|---|---:|
| Driver | GTiff |
| Width | 1,125 |
| Height | 1,239 |
| Bands | 4 |
| CRS | EPSG:4326 |
| Data type | uint8 |
| Nodata | null |

The file includes geographic bounds in longitude/latitude coordinates:

```text
left: -95.73131385948028
bottom: 29.703945043310878
right: -95.72828209817133
top: 29.707284023099135

The corresponding building damage assessment JSON file contained 4 annotation records. Each annotation record included:

building_id
view_id
label
pixels
boundary
source
filename

The pixels field contained polygon points in image pixel coordinates, which can be used to extract building-level crops from the GeoTIFF.

Crop extraction demonstration

A building-crop extraction script was added:

scripts/extract_crasar_building_crops.py

This script takes:

GeoTIFF image + building damage assessment JSON

and outputs:

building crop PNG files + metadata CSV

For the inspected Hurricane Harvey CREWED RGB GeoTIFF, the script extracted 4 building crops. The resulting metadata CSV preserved:

crop_path
building_id
view_id
label
source
filename
original_geotiff_path
pixel bounds
CRS
resolution
pHash
quality metrics

This demonstrates that georeferenced GeoTIFF data can be converted into ordinary image crops for computer vision models while preserving the original building-level and geospatial metadata.

Important interpretation

A GeoTIFF can be converted into normal image data, but it is not originally the same as a video frame.

A video frame is usually just an RGB image in a temporal sequence.

A GeoTIFF is a georeferenced raster. It contains image bands plus spatial metadata such as CRS, transform, bounds, and resolution. This allows building annotations to be aligned with the image and enables building-level crop extraction.

Therefore, if the project only wants non-georeferenced image/frame-like data, the correct approach is:

GeoTIFF + annotation JSON -> building crop PNG/JPG files + metadata CSV
