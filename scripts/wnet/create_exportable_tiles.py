from scripts.common.create_exportable_tiles import create_exportable_tile_table

if __name__ == '__main__':
    create_exportable_tile_table(table_suffix="wnet", tile_size=224.0, label_size=224.0)
