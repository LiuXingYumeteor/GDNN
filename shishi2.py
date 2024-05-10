from typing import List
import math


def min_skips(dist: List[int], speed: int, hours_before: int) -> int:
    dp = [[math.inf] * (len(dist) + 1) for _ in range(len(dist) + 1)]
    dp[0][0] = 0

    for i in range(1, len(dist) + 1):
        for j in range(i + 1):
            if j < i:
                dp[i][j] = math.ceil(dp[i - 1][j] / speed) * speed + dist[i - 1]
            if j > 0:
                dp[i][j] = dp[i - 1][j - 1] + dist[i - 1]

    # The answer will be the smallest number of skips such that dp[n][skips] / speed <= hours_before
    for skips in range(len(dist) + 1):
        if dp[len(dist)][skips] <= hours_before * speed:
            return skips

    return -1


# Example usage:
dist = [5, 3, 2]  # Example distances for each road
speed = 5  # Example speed in km/h
hours_before = 6  # Example deadline in hours
min_skips(dist, speed, hours_before)
