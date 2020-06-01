CREATE INDEX ON public.dop_rgb USING gist (st_convexhull(rast));
CREATE INDEX ON public.dop_nir USING gist (st_convexhull(rast));

SELECT AddRasterConstraints('public','dop_rgb','rast',TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,FALSE,TRUE,TRUE,TRUE,TRUE,TRUE);
SELECT AddRasterConstraints('public','dop_nir','rast',TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,FALSE,TRUE,TRUE,TRUE,TRUE,TRUE);