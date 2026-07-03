# CRASAR Hurricane GeoTIFF Pipeline Summary

## Purpose

The purpose of this phase was to compare georeferenced CRASAR `.geo.tif` data with ordinary non-georeferenced video/frame data, identify hurricane-related CRASAR files, and demonstrate how GeoTIFF imagery can be converted into ordinary image crops for computer vision.

## Hurricane-related file filtering

A filename/metadata-based filter was created to identify hurricane-related CRASAR-U-DROIDs files.

The filter found:

| Category | Count |
|---|---:|
| Total hurricane-related files | 255 |
| Hurricane imagery GeoTIFF files | 73 |
| Hurricane annotation JSON files | 182 |

The matched hurricane subsets were:

| Hurricane | Files |
|---|---:|
| Hurricane Michael | 51 |
| Hurricane Idalia | 50 |
| Hurricane Harvey | 61 |
| Hurricane Ian | 93 |

The hurricane-related files include multiple sources:

| Source | Files |
|---|---:|
| CREWED | 51 |
| SATELLITE | 140 |
| UAS | 60 |
| UAS_DSM | 4 |

## GeoTIFF versus ordinary frames

Ordinary video/frame data is usually RGB image data in a temporal sequence. A normal frame does not usually include spatial metadata.

A GeoTIFF is different. It is a georeferenced raster image. It contains pixel data plus spatial metadata such as:

```text
CRS
affine transform
spatial bounds
resolution
number of bands
data type

This means that a GeoTIFF image can be connected to real-world coordinates, while an ordinary video frame usually cannot.

Inspected CREWED GeoTIFF example

The following hurricane-related GeoTIFF was downloaded and inspected:

train/imagery/CREWED/090401-DMS-Assessment-Westpark.geo.tif_20170830_RGB.geo.tif

This file is related to Hurricane Harvey.

The GeoTIFF metadata showed:

| Field      |                         Value |
| ---------- | ----------------------------: |
| Driver     |                         GTiff |
| Width      |                         1,125 |
| Height     |                         1,239 |
| Bands      |                             4 |
| CRS        |                     EPSG:4326 |
| Data type  |                         uint8 |
| Nodata     |                          null |
| Resolution | 2.694898941e-06 degrees/pixel |

The corresponding building damage assessment JSON file contained 4 annotation records. Each record included:

building_id
view_id
label
pixels
boundary
source
filename

The pixels field contained building polygon points in image pixel coordinates.

Inspected SATELLITE GeoTIFF example

The following hurricane-related SATELLITE GeoTIFF was also downloaded and inspected:

train/imagery/SATELLITE/090401-DMS-Assessment-Westpark.geo.tif_103001006F884000.tif.geo.tif

This file is also related to Hurricane Harvey.

The GeoTIFF metadata showed:

| Field      |                         Value |
| ---------- | ----------------------------: |
| Driver     |                         GTiff |
| Width      |                           420 |
| Height     |                           463 |
| Bands      |                             3 |
| CRS        |                     EPSG:4326 |
| Data type  |                         uint8 |
| Nodata     |                          null |
| Resolution | 7.219086202e-06 degrees/pixel |

This SATELLITE file is smaller and lower-resolution than the inspected CREWED example. It also demonstrates that CRASAR contains different imagery sources with different spatial resolutions and raster structures.

Converting GeoTIFF to ordinary image data

A script was added to export a normal RGB PNG preview from a GeoTIFF:

scripts/export_geotiff_rgb_preview.py

This demonstrates that if the project only wants non-georeferenced image/frame-like data, a GeoTIFF can be converted into a PNG/JPG-style image.

However, the geospatial metadata should be preserved separately in a CSV or JSON file.

Building crop extraction

A script was added to extract building-level crops from CRASAR GeoTIFF imagery:

scripts/extract_crasar_building_crops.py

The script takes:

GeoTIFF image + building_damage_assessment JSON

and outputs:

building crop PNG files + metadata CSV

The metadata CSV preserves:

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

This demonstrates the complete conversion:

georeferenced GeoTIFF + annotation JSON
→ building-level PNG crops
→ ordinary computer-vision input data
Connection to pHash

For DoriaNET-style UAV video, pHash is useful for detecting temporal redundancy between consecutive frames.

For CRASAR GeoTIFF data, pHash should be applied after extracting image tiles or building crops. It can then be used to:

detect visually redundant building crops,
audit near-duplicate images,
select representative crops,
support leakage-resistant grouped splits by building_id.
Important conclusion

CRASAR GeoTIFF data is not the same as video frame data.

The correct conversion pathway is:

GeoTIFF raster + annotation polygons
→ RGB image tiles or building crops
→ PNG/JPG files for computer vision
→ metadata CSV preserving building IDs, labels, crop bounds, and geospatial information

This allows the project to use standard image-based machine learning models while still preserving the structure and metadata advantages of georeferenced imagery.

## Visual quality limitation from the first crop demos

The initial CREWED and SATELLITE crop extraction tests demonstrate that the technical pipeline works, but they also show that not all GeoTIFF-derived crops are visually useful.

Some extracted crops contain large black regions. These likely correspond to no-data/background areas or regions outside the useful raster footprint. This is different from ordinary video frames, where the whole frame is usually a valid RGB image.

The SATELLITE example was especially limited because it was very small, 420 x 463 pixels, and many extracted crops were labelled `obscured` or `un-classified`. These labels are not ideal for a main supervised damage-classification task.

Therefore, the next step should not be to process all hurricane GeoTIFFs blindly. The next step should be to rank hurricane-related GeoTIFF/annotation pairs by:

1. manageable file size,
2. number of useful building annotations,
3. label distribution,
4. source type,
5. expected crop quality,
6. proportion of labels in the main damage classes.

For the main classification task, useful labels are likely:

```text
no damage
minor damage
major damage
destroyed

The labels un-classified and obscured should probably be excluded from the first clean supervised setup or handled separately.

## Candidate ranking result

The first small GeoTIFF demos were useful for proving the pipeline but were not visually ideal for training. The SATELLITE demo in particular showed low-resolution crops with substantial black no-data/background regions and mostly `obscured` or `un-classified` labels.

To avoid processing files blindly, a ranking script was added:

```text
scripts/rank_crasar_hurricane_crop_candidates.py

This script ranks hurricane-related GeoTIFF/annotation pairs using file size, number of useful labels, useful-label rate, source, and label distribution.

For CREWED imagery under 150 MB, the best candidate was:

train/imagery/CREWED/090403-Lancaster-Canyon-Gate.geo.tif_20170830_RGB.geo.tif

Its corresponding annotation file contained:
| Metric               |    Value |
| -------------------- | -------: |
| Annotation records   |      647 |
| Useful records       |      629 |
| Useful rate          |    0.972 |
| Excluded records     |       18 |
| Major damage records |      629 |
| File size            | 60.83 MB |

This is a much better candidate for building-crop extraction than the first small Westpark example, which had only 4 annotation records and only 1 useful label.
