def generate_n_shapelets_per_size(
    ts_length,
    n_shapelets_per_length=4,
    min_length=8,
    start_divider=2,
    divider_multiplier=2,
):
    n_shapelets_per_size = dict()
    while int(ts_length / start_divider) >= min_length:
        n_shapelets_per_size[int(ts_length / start_divider)] = n_shapelets_per_length
        start_divider *= divider_multiplier
    return n_shapelets_per_size
