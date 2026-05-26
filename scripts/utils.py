def ranges_to_indices(ranges: str) -> list[int]:
    indices: set[int] = set()
    groups = ranges.split(",")
    for g in groups:
        rs = g.split("-")
        if len(rs) == 1:
            indices.add(int(g))
            continue
        assert len(rs) == 2
        start_idx, end_idx = rs
        for i in range(int(start_idx), int(end_idx) + 1):
            indices.add(i)
    return sorted(indices)
