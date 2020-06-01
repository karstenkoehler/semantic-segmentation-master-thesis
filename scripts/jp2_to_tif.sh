#!/bin/bash
# convert JPEG 2000 files to the GeoTiff format

for in_file in $HOME/thesis/dop_jp2/*.jp2; do
  file_name=${in_file##*/}
  file_name=${file_name%.*}
  echo "translating $file_name.jp2 to $file_name.tif ..."
  gdal_translate -of GTiff "$in_file" "$HOME/thesis/dop_tif/$file_name.tif"
done