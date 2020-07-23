# connection string for database
POSTGRES_CONNECTION_DSN = "dbname='dop10rgbi_nrw' user='postgres' host='localhost' password='root'"

# grayscale values for the segmentation categories
LABEL_GRAYSCALE_VALUES = [
    62,  # buildings
    104,  # water
    118,  # forest
    193,  # traffic
    200,  # urban greens
    226,  # agriculture
]

# color values for the segmentation categories
LABEL_RGB_VALUES = [
    (3, 0, 208),  # buildings
    (240, 126, 11),  # water
    (40, 171, 44),  # forest
    (193, 193, 193),  # traffic
    (39, 255, 154),  # urban greens
    (132, 240, 235),  # agriculture
]
