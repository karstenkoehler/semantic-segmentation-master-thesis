from scripts.common.create_exportable_tiles import create_exportable_tile_table

if __name__ == '__main__':
    create_exportable_tile_table(table_suffix="densenet", tile_size=256.0, label_size=256.0)
