#!/bin/bash
# import GeoTiff files to PostgreSQL databases

for in_file in $HOME/thesis/dop_tif/*.tif; do

  # -t 250x250  tile size
  # -b 1-3      insert raster bands 1, 2 and 3
  # -F          add column with filename
  # -s 25832    set SRID
  # -e          don't use transactions
  # -a          append mode (don't create the table)
  raster2pgsql -t 250x250 -b 1-3 -F -I -s 25832 -e -a "$in_file" public.dop_rgb | psql -d dop10rgbi_nrw -q
  raster2pgsql -t 250x250 -b 4 -F -I -s 25832 -e -a "$in_file" public.dop_nir | psql -d dop10rgbi_nrw -q
done