import psycopg2
import numpy as np

from scripts.common.constants import POSTGRES_CONNECTION_DSN, LABEL_GRAYSCALE_VALUES


def chunks(L, n):
    """Splits list L in chunks with size n."""
    for i in range(0, len(L), n):
        yield L[i:i + n]


def get_training_gids_from_database(table_suffix):
    """Reads all GIDS from respective table and returns them as a list."""
    with psycopg2.connect(POSTGRES_CONNECTION_DSN) as db:
        with db.cursor() as cur:
            stmt = f"SELECT gid FROM geom_tiles_{table_suffix} WHERE NOT test_set;;"
            cur.execute(stmt)
            return [int(row[0]) for row in cur.fetchall()]


def get_training_gids_from_file(file_path):
    """Reads GIDS from a file and returns them as a list."""
    with open(file_path, 'r') as f:
        return [int(line) for line in f.read().splitlines()]


def one_hot_encoding(label):
    encoded = []
    for val in LABEL_GRAYSCALE_VALUES:
        encoded.append((label == val) * 1.0)
    return np.stack(encoded, axis=2)
