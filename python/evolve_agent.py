"""Small, shared Python primitives for the live 2048 learning server."""

import math
from random import Random, choice, gauss, random
from typing import Iterable

ACTIONS = ["up", "right", "down", "left"]
FEATURES = ["empty", "largest", "corner", "smoothness", "merges", "monotonicity"]
# Zero means the first generation has no hand-tuned strategy.
STARTING_WEIGHTS = [0.0] * len(FEATURES)


def slide_line(line: list[int]) -> tuple[list[int], int]:
    compact = [value for value in line if value]
    result: list[int] = []
    points = 0
    index = 0
    while index < len(compact):
        if index + 1 < len(compact) and compact[index] == compact[index + 1]:
            merged = compact[index] * 2
            result.append(merged)
            points += merged
            index += 2
        else:
            result.append(compact[index])
            index += 1
    return result + [0] * (4 - len(result)), points


def move_board(board: list[list[int]], action: str) -> tuple[list[list[int]], int, bool]:
    result = [[0] * 4 for _ in range(4)]
    total_points = 0

    def read_line(index: int) -> list[int]:
        if action == "left":
            return board[index][:]
        if action == "right":
            return board[index][::-1]
        if action == "up":
            return [board[row][index] for row in range(4)]
        return [board[row][index] for row in range(3, -1, -1)]

    def write_line(index: int, line: list[int]) -> None:
        if action == "left":
            result[index] = line
        elif action == "right":
            result[index] = line[::-1]
        elif action == "up":
            for row in range(4):
                result[row][index] = line[row]
        else:
            for row in range(4):
                result[3 - row][index] = line[row]

    for index in range(4):
        line, points = slide_line(read_line(index))
        write_line(index, line)
        total_points += points
    return result, total_points, result != board


def spawn_tile(board: list[list[int]], rng: Random) -> bool:
    empty = [(row, column) for row in range(4) for column in range(4) if board[row][column] == 0]
    if not empty:
        return False
    row, column = rng.choice(empty)
    board[row][column] = 4 if rng.random() < 0.1 else 2
    return True


def board_features(board: list[list[int]]) -> list[float]:
    logs = [[math.log2(value) if value else 0.0 for value in row] for row in board]
    empty = sum(value == 0 for row in board for value in row) / 16.0
    largest_log = max(value for row in logs for value in row)
    largest = largest_log / 16.0
    largest_tile = 2 ** int(largest_log)
    corner = float(largest_tile in (board[0][0], board[0][3], board[3][0], board[3][3]))

    smoothness = 0.0
    merges = 0.0
    for row in range(4):
        for column in range(3):
            if logs[row][column] and logs[row][column + 1]:
                smoothness -= abs(logs[row][column] - logs[row][column + 1]) / 16.0
            if board[row][column] and board[row][column] == board[row][column + 1]:
                merges += 0.25
    for column in range(4):
        for row in range(3):
            if logs[row][column] and logs[row + 1][column]:
                smoothness -= abs(logs[row][column] - logs[row + 1][column]) / 16.0
            if board[row][column] and board[row][column] == board[row + 1][column]:
                merges += 0.25

    lines = logs + [[logs[row][column] for row in range(4)] for column in range(4)]
    monotonicity = 0.0
    for line in lines:
        increasing = sum(max(0.0, right - left) for left, right in zip(line, line[1:]))
        decreasing = sum(max(0.0, left - right) for left, right in zip(line, line[1:]))
        monotonicity += max(increasing, decreasing) / 16.0
    return [empty, largest, corner, smoothness, merges, monotonicity]


def policy(board: list[list[int]], weights: list[float]) -> str:
    candidates = []
    for action in ACTIONS:
        next_board, _, changed = move_board(board, action)
        if changed:
            value = sum(feature * weight for feature, weight in zip(board_features(next_board), weights))
            candidates.append((value, action))
    return max(candidates)[1] if candidates else choice(ACTIONS)


def breed(survivors: list[list[float]], mutation_amount: float) -> list[float]:
    first, second = choice(survivors), choice(survivors)
    child = [first[index] if random() < 0.5 else second[index] for index in range(len(FEATURES))]
    return [weight + gauss(0, mutation_amount) for weight in child]
