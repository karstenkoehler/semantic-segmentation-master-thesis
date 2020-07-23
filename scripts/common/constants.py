# connection string for database
POSTGRES_CONNECTION_DSN = "dbname='dop10rgbi_nrw' user='postgres' host='localhost' password='root'"

# grayscale values for the segmentation categories
LABEL_GRAYSCALE_VALUES = [
    62,  # buildings
    104,  # water
    118,  # forest
    193,  # traffic
    200,  # urban greens
    226  # agriculture
]