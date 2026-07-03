# Meeting Update: GeoTIFF vs Frame Data and Hurricane Data Extraction

## 1. Main question

The current task was to understand:

```text
What is inside CRASAR .geo.tif data?
How is it different from ordinary video/frame data?
How can we extract ordinary image/frame-like data from GeoTIFFs?
How can we isolate hurricane-related CRASAR data?
2. Difference between GeoTIFF and ordinary frames

Ordinary video frames are usually just RGB images in a temporal sequence.

A GeoTIFF is different. It is a georeferenced raster image. It contains pixel data plus spatial metadata such as:

CRS
affine transform
spatial bounds
resolution
number of bands
data type

This means GeoTIFF pixels can be connected to real-world coordinates, while ordinary video frames usually cannot.

3. Hurricane-related CRASAR filtering

A filename/metadata-based filter was created to isolate hurricane-related CRASAR files.

The filter found:

Category	Count
Total hurricane-related files	255
Hurricane imagery GeoTIFF files	73
Hurricane annotation JSON files	182

The matched hurricanes were:

Hurricane	Files
Hurricane Michael	51
Hurricane Idalia	50
Hurricane Harvey	61
Hurricane Ian	93

The files include CREWED, SATELLITE, UAS, and UAS_DSM sources.

4. GeoTIFF inspection

Two hurricane-related GeoTIFFs were inspected.

CREWED example
train/imagery/CREWED/090401-DMS-Assessment-Westpark.geo.tif_20170830_RGB.geo.tif

Metadata:

Field	Value
Width	1,125
Height	1,239
Bands	4
CRS	EPSG:4326
SATELLITE example
train/imagery/SATELLITE/090401-DMS-Assessment-Westpark.geo.tif_103001006F884000.tif.geo.tif

Metadata:

Field	Value
Width	420
Height	463
Bands	3
CRS	EPSG:4326

This showed that CRASAR source types can have different sizes, bands, and spatial resolutions.

5. Converting GeoTIFF to ordinary image data

A script was created to export GeoTIFFs as normal RGB PNG previews:

scripts/export_geotiff_rgb_preview.py

This demonstrates that GeoTIFFs can be converted into frame-like image files.

However, the metadata should be preserved separately.

6. Building crop extraction

A building-crop extraction script was created:

scripts/extract_crasar_building_crops.py

It takes:

GeoTIFF image + building_damage_assessment JSON

and outputs:

building crop PNG files + metadata CSV

The annotation JSON provides:

building_id
view_id
label
pixels
boundary
source
filename

The pixels field gives polygon points in image coordinates, so it can be used to extract building-level crops.

7. Candidate ranking

The first small GeoTIFF tests proved the pipeline but produced poor crops. This showed that file selection matters.

A ranking script was created:

scripts/rank_crasar_hurricane_crop_candidates.py

It ranks hurricane-related GeoTIFF/annotation pairs by:

file size
source
number of useful labels
useful-label rate
label distribution
8. Good candidate result

The best CREWED candidate under 150 MB was:

train/imagery/CREWED/090403-Lancaster-Canyon-Gate.geo.tif_20170830_RGB.geo.tif

Its annotation file contained:

Metric	Value
Annotation records	647
Useful records	629
Useful-label rate	0.972
Major damage records	629
Un-classified records	18
File size	60.83 MB

The crop extraction pipeline produced 647 crops.

9. Crop filtering result

A crop-quality filtering script was added:

scripts/filter_extracted_crops.py

For Lancaster-Canyon-Gate:

Metric	Value
Input crops	647
Filtered crops kept	442
Kept rate	0.6832
Final useful crops	442 major damage

Removed crops included:

176 low-quality crops
18 excluded-label crops
29 crops with too much black background
10. Main conclusion

CRASAR GeoTIFF data is not the same as ordinary video-frame data.

The correct conversion pathway is:

GeoTIFF raster + annotation polygons
→ RGB image tiles or building crops
→ PNG/JPG files for computer vision
→ metadata CSV preserving building IDs, labels, crop bounds, and geospatial information

The pipeline works, but candidate selection and crop-quality filtering are necessary before building a training dataset.

11. Next step

The next step should be to test a more class-diverse hurricane candidate, not only a major-damage-heavy file.

A good next candidate is:

train/imagery/SATELLITE/1002-Ft-Myers-Beach-TFD.geo.tif_10300100DB06A700-visual.tif.geo.tif

Its annotation distribution is:

Label	Count
no damage	330
minor damage	246
major damage	40
destroyed	147
un-classified	7
obscured	0

This would be better for testing a multi-class damage-classification setup.
