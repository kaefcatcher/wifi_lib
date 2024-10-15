from typing import List


def pad_data(data: List[complex], target_length: int = 52) -> List[complex]:
    if target_length > len(data):
        return data + [0+0j] * (target_length - len(data))
    return data
