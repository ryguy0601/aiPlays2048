"""Multi-paradigm Machine Learning architectures for 2048:
1. Deep Neuroevolution MLP (Feedforward Net + GA)
2. Q-Learning with Linear Function Approximation (TD-Learning)
3. N-Tuple Network (Temporal Difference with Pattern Lookups)
"""

from random import Random, choice, gauss, random, sample
from typing import Sequence, Any

ACTIONS = ["up", "right", "down", "left"]
ML_MODELS = ["mlp_neuroevolution", "q_learning_linear", "n_tuple_network"]

# Precomputed log2 table for fast board lookups
_LOG2_TABLE: dict[int, int] = {0: 0}
for _p in range(1, 18):
    _LOG2_TABLE[1 << _p] = _p

def get_log(val: int) -> int:
    return _LOG2_TABLE.get(val, int(val.bit_length() - 1) if val else 0)


# ==========================================
# 1. DEEP NEUROEVOLUTION (MLP)
# ==========================================

LAYER_SIZES = [16, 32, 16, 1]

def relu(x: float) -> float:
    return x if x > 0.0 else 0.0

class MLPModel:
    def __init__(self, weights=None, biases=None):
        if weights is not None and biases is not None:
            self.weights = weights
            self.biases = biases
        else:
            self.weights, self.biases = self._init_weights()

    @staticmethod
    def _init_weights():
        w, b = [], []
        for i in range(len(LAYER_SIZES) - 1):
            fan_in = LAYER_SIZES[i]
            fan_out = LAYER_SIZES[i + 1]
            scale = (2.0 / fan_in) ** 0.5
            w.append([[gauss(0.0, scale) for _ in range(fan_out)] for _ in range(fan_in)])
            b.append([0.0 for _ in range(fan_out)])
        return w, b

    def forward(self, inputs: Sequence[float]) -> float:
        acts = list(inputs)
        num_layers = len(self.weights)
        for l_idx in range(num_layers):
            w = self.weights[l_idx]
            b = self.biases[l_idx]
            fan_in, fan_out = len(w), len(b)
            next_act = [0.0] * fan_out
            for j in range(fan_out):
                tot = b[j]
                for i in range(fan_in):
                    tot += acts[i] * w[i][j]
                next_act[j] = relu(tot) if l_idx < num_layers - 1 else tot
            acts = next_act
        return acts[0]

    def evaluate(self, board: list[list[int]]) -> float:
        nn_in = [get_log(board[r][c]) / 16.0 for r in range(4) for c in range(4)]
        return self.forward(nn_in)

    def clone(self) -> "MLPModel":
        w = [[[val for val in row] for row in l] for l in self.weights]
        b = [[val for val in l] for l in self.biases]
        return MLPModel(w, b)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "mlp_neuroevolution",
            "weights": self.weights,
            "biases": self.biases,
            "layers": LAYER_SIZES,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MLPModel":
        return cls(weights=data["weights"], biases=data["biases"])

def breed_mlp(p1: MLPModel, p2: MLPModel, mutation_rate: float, mutation_strength: float) -> MLPModel:
    child_w, child_b = [], []
    for l_idx in range(len(p1.weights)):
        w1, w2 = p1.weights[l_idx], p2.weights[l_idx]
        b1, b2 = p1.biases[l_idx], p2.biases[l_idx]
        layer_w = []
        for i in range(len(w1)):
            row = []
            for j in range(len(w1[0])):
                g = w1[i][j] if random() < 0.5 else w2[i][j]
                if random() < mutation_rate:
                    g += gauss(0.0, mutation_strength)
                row.append(g)
            layer_w.append(row)
        child_w.append(layer_w)

        layer_b = []
        for j in range(len(b1)):
            bg = b1[j] if random() < 0.5 else b2[j]
            if random() < mutation_rate:
                bg += gauss(0.0, mutation_strength)
            layer_b.append(bg)
        child_b.append(layer_b)
    return MLPModel(child_w, child_b)


# ==========================================
# 2. Q-LEARNING WITH LINEAR APPROXIMATION
# ==========================================

class QLearningLinearModel:
    FEATURES = ["empty", "max_tile", "smoothness", "monotonicity", "snake"]

    def __init__(self, weights=None, lr=0.01, gamma=0.95, epsilon=0.05):
        self.weights = list(weights) if weights else [0.0] * len(self.FEATURES)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)

    @staticmethod
    def extract_features(board: list[list[int]]) -> list[float]:
        empty = sum(cell == 0 for row in board for cell in row) / 16.0
        logs = [[get_log(c) for c in row] for row in board]
        max_log = max(max(r) for r in logs) / 16.0

        smoothness = 0.0
        for r in range(4):
            for c in range(3):
                smoothness -= abs(logs[r][c] - logs[r][c + 1])
                smoothness -= abs(logs[c][r] - logs[c + 1][r])
        smoothness /= 48.0

        mono = 0.0
        for r in range(4):
            diffs = [logs[r][c] - logs[r][c + 1] for c in range(3)]
            mono -= min(sum(d for d in diffs if d > 0), sum(-d for d in diffs if d < 0))
        mono /= 24.0

        # Snake matrix gradient
        snake_matrix = [[15, 14, 13, 12], [8, 9, 10, 11], [7, 6, 5, 4], [0, 1, 2, 3]]
        snake = sum(logs[r][c] * snake_matrix[r][c] for r in range(4) for c in range(4)) / 240.0

        return [empty, max_log, smoothness, mono, snake]

    def evaluate(self, board: list[list[int]]) -> float:
        phi = self.extract_features(board)
        return sum(f * w for f, w in zip(phi, self.weights))

    def update_td(self, state_features: list[float], reward: float, next_state_value: float):
        pred = sum(f * w for f, w in zip(state_features, self.weights))
        target = reward + self.gamma * next_state_value
        error = target - pred
        for i in range(len(self.weights)):
            self.weights[i] += self.lr * error * state_features[i]

    def clone(self) -> "QLearningLinearModel":
        return QLearningLinearModel(self.weights[:], self.lr, self.gamma, self.epsilon)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "q_learning_linear",
            "weights": self.weights,
            "params": {"lr": self.lr, "gamma": self.gamma, "epsilon": self.epsilon},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QLearningLinearModel":
        p = data.get("params", {})
        return cls(
            weights=data["weights"],
            lr=p.get("lr", 0.01),
            gamma=p.get("gamma", 0.95),
            epsilon=p.get("epsilon", 0.05),
        )


# ==========================================
# 3. N-TUPLE NETWORK (TD PATTERN TABLES)
# ==========================================

# 4-tuples covering horizontal rows, vertical cols, and 2x2 squares
TUPLES = [
    (0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15),  # rows
    (0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15),  # cols
    (0, 1, 4, 5), (1, 2, 5, 6), (2, 3, 6, 7), (4, 5, 8, 9), (5, 6, 9, 10), (6, 7, 10, 11)  # squares
]

class NTupleNetwork:
    def __init__(self, tables=None, lr=0.02):
        self.lr = float(lr)
        # Convert incoming keys (whether str from JSON or int) into Python integers
        if tables is not None:
            self.tables = [{int(k): float(v) for k, v in t.items()} for t in tables]
        else:
            self.tables = [{} for _ in range(len(TUPLES))]

    @staticmethod
    def _tuple_index(board_flat: list[int], indices: tuple[int, ...]) -> int:
        idx = 0
        for pos in indices:
            idx = (idx << 4) | (get_log(board_flat[pos]) & 0xF)
        return idx

    def evaluate(self, board: list[list[int]]) -> float:
        flat = [board[r][c] for r in range(4) for c in range(4)]
        total = 0.0
        for t_idx, tup in enumerate(TUPLES):
            idx = self._tuple_index(flat, tup)
            total += self.tables[t_idx].get(idx, 0.0)
        return total

    def update_td(self, board: list[list[int]], reward: float, next_val: float):
        flat = [board[r][c] for r in range(4) for c in range(4)]
        cur_val = self.evaluate(board)
        delta = (reward + next_val) - cur_val
        step = (self.lr / len(TUPLES)) * delta

        for t_idx, tup in enumerate(TUPLES):
            idx = self._tuple_index(flat, tup)
            self.tables[t_idx][idx] = self.tables[t_idx].get(idx, 0.0) + step

    def clone(self) -> "NTupleNetwork":
        cloned_tables = [{k: v for k, v in t.items()} for t in self.tables]
        return NTupleNetwork(cloned_tables, self.lr)

    def to_dict(self) -> dict[str, Any]:
        # JSON specification requires all dictionary keys to be strings
        stringified_tables = [
            {str(k): float(v) for k, v in t.items()}
            for t in self.tables
        ]
        return {
            "type": "n_tuple_network",
            "params": {"lr": self.lr},
            "tables": stringified_tables,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NTupleNetwork":
        p = data.get("params", {})
        return cls(tables=data.get("tables", []), lr=p.get("lr", 0.02))


# ==========================================
# SIMULATION & ACTION SELECTION
# ==========================================

def slide_line(line: Sequence[int]) -> tuple[list[int], int]:
    compact = [v for v in line if v]
    length = len(compact)
    if length <= 1:
        return (compact + [0] * (4 - length)), 0
    res, pts, i = [], 0, 0
    while i < length:
        val = compact[i]
        if i + 1 < length and val == compact[i + 1]:
            m = val << 1
            res.append(m)
            pts += m
            i += 2
        else:
            res.append(val)
            i += 1
    while len(res) < 4:
        res.append(0)
    return res, pts

def move_board(board: list[list[int]], action: str) -> tuple[list[list[int]], int, bool]:
    b0, b1, b2, b3 = board
    if action == "left":
        r0, p0 = slide_line(b0); r1, p1 = slide_line(b1); r2, p2 = slide_line(b2); r3, p3 = slide_line(b3)
        res = [r0, r1, r2, r3]
    elif action == "right":
        r0, p0 = slide_line((b0[3], b0[2], b0[1], b0[0]))
        r1, p1 = slide_line((b1[3], b1[2], b1[1], b1[0]))
        r2, p2 = slide_line((b2[3], b2[2], b2[1], b2[0]))
        r3, p3 = slide_line((b3[3], b3[2], b3[1], b3[0]))
        res = [r0[::-1], r1[::-1], r2[::-1], r3[::-1]]
    elif action == "up":
        c0, p0 = slide_line((b0[0], b1[0], b2[0], b3[0]))
        c1, p1 = slide_line((b0[1], b1[1], b2[1], b3[1]))
        c2, p2 = slide_line((b0[2], b1[2], b2[2], b3[2]))
        c3, p3 = slide_line((b0[3], b1[3], b2[3], b3[3]))
        res = [[c0[0], c1[0], c2[0], c3[0]], [c0[1], c1[1], c2[1], c3[1]], [c0[2], c1[2], c2[2], c3[2]], [c0[3], c1[3], c2[3], c3[3]]]
    else:
        c0, p0 = slide_line((b3[0], b2[0], b1[0], b0[0]))
        c1, p1 = slide_line((b3[1], b2[1], b1[1], b0[1]))
        c2, p2 = slide_line((b3[2], b2[2], b1[2], b0[2]))
        c3, p3 = slide_line((b3[3], b2[3], b1[3], b0[3]))
        res = [[c0[3], c1[3], c2[3], c3[3]], [c0[2], c1[2], c2[2], c3[2]], [c0[1], c1[1], c2[1], c3[1]], [c0[0], c1[0], c2[0], c3[0]]]
    return res, (p0 + p1 + p2 + p3), res != board

def spawn_tile(board: list[list[int]], rng: Random) -> bool:
    empty = [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]
    if not empty:
        return False
    row, col = rng.choice(empty)
    board[row][col] = 4 if rng.random() < 0.1 else 2
    return True

def select_action(board: list[list[int]], model: Any) -> tuple[str, float]:
    """1-step lookahead evaluation for any ML model instance."""
    best_score = -float("inf")
    best_action = None

    # Epsilon-greedy exploration for Q-Learning
    if getattr(model, "epsilon", 0) > 0 and random() < model.epsilon:
        valid_actions = [a for a in ACTIONS if move_board(board, a)[2]]
        return (choice(valid_actions) if valid_actions else "up"), 0.0

    for action in ACTIONS:
        next_b, pts, changed = move_board(board, action)
        if not changed:
            continue
        v = model.evaluate(next_b) + pts
        if v > best_score:
            best_score = v
            best_action = action

    return (best_action if best_action else choice(ACTIONS)), (best_score if best_score != -float("inf") else 0.0)