# Georeferenced GeoTIFF Data vs Non-Georeferenced Video/Frame Data

## Purpose

This document summarizes the difference between georeferenced `.geo.tif` data and ordinary non-georeferenced video/frame data for the building damage assessment project.

## Non-georeferenced video or frame data

Ordinary UAV video data is usually represented as a sequence of RGB frames:

```text
frame_000001.jpg
frame_000002.jpg
frame_000003.jpg

These frames contain visual information but usually do not contain spatial metadata. A pixel location only refers to a position inside the image, not to a real-world coordinate.

For example:
pixel (x=300, y=500)
only means that the pixel is located at that position in the image array.

This type of data is useful for standard computer vision pipelines, but it has several limitations:

Frames may be highly redundant because they come from continuous video.
The same building may appear in many nearly identical frames.
Without building-level annotations, it is difficult to know whether the same building appears in train, validation, and test sets.
Building crops must be obtained through manual annotation, object detection, segmentation, or external metadata.

Georeferenced GeoTIFF data

A .geo.tif file is a raster image with geospatial metadata. It contains image pixels, but also information that maps pixels to real-world coordinates.

A GeoTIFF usually includes:

Image width and height.
Number of bands.
Coordinate reference system, or CRS.
Affine transform from pixel coordinates to map coordinates.
Spatial bounds.
Spatial resolution.
Pixel data.

This means that a pixel location can be related to a real-world location.

For example:
pixel (x=300, y=500) -> map coordinate / geographic position
This is important because building polygons can be aligned with the imagery. In CRASAR-U-DROIDs, annotation records include fields such as:
building_id
view_id
label
pixels
boundary
source
filename

These fields make it possible to extract building-level crops and preserve building identifiers and labels.

Main difference

The main difference is that ordinary video frames are just images, while GeoTIFF files are images with spatial reference.

| Aspect                   | Video/frame data              | GeoTIFF data                          |
| ------------------------ | ----------------------------- | ------------------------------------- |
| Data type                | Sequential RGB frames         | Georeferenced raster imagery          |
| Temporal structure       | Yes, if from video            | Usually no                            |
| Spatial metadata         | Usually no                    | Yes                                   |
| CRS                      | No                            | Yes                                   |
| Pixel-to-world mapping   | No                            | Yes                                   |
| Building polygons        | Not inherent                  | Can be aligned                        |
| Building crop extraction | Requires detection/annotation | Can use polygons or pixel boundaries  |
| Risk of redundancy       | Temporal frame redundancy     | Cross-file/source/building repetition |
| Typical output for CNN   | Frames or crops               | Exported tiles or building crops      |

How to use GeoTIFF as ordinary image/frame data

If we want to use GeoTIFF data in a standard computer vision model, we can export image tiles or building crops as normal PNG/JPG files.

The conversion is:

GeoTIFF orthomosaic -> RGB image tiles or building crops -> PNG/JPG files

Once exported, the image can be treated like a normal frame. However, we should preserve the geospatial and building metadata in a CSV file.

A useful metadata table would include:

crop_path
building_id
label
source
filename
original_geotiff
pixel_bounds
crs
transform

Implication for this project

For DoriaNET-style UAV video, pHash is useful for detecting temporal redundancy between consecutive frames.

For CRASAR-U-DROIDs, the main issue is different. CRASAR is not organized as raw video. It is organized around georeferenced imagery and building-level annotations. Therefore, pHash should be applied after extracting building crops or image tiles, not directly as a frame-sequence method.

The next methodological step is to inspect the GeoTIFF metadata, filter hurricane-related files, and decide whether to export:

full RGB images,
fixed-size tiles, or
building-level crops using polygon/pixel annotations.
