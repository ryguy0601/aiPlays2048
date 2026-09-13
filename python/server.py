"""Python-owned 2048 learning server for the browser arena.

The browser is only a view. This process owns game state, rewards, demerits,
policy selection, global-weight learning, generations, and population size.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from random import Random
from typing import Any

from evolve_agent import ACTIONS, FEATURES, STARTING_WEIGHTS, breed, move_board, policy, spawn_tile

HOST = "127.0.0.1"
PORT = 8765
TICK_SECONDS = 0.115


class PythonGame:
    def __init__(self, agent_id: int, seed: int) -> None:
        self.agent_id = agent_id
        self.reset(seed)

    def reset(self, seed: int) -> None:
        self.rng = Random(seed)
        self.board = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.moves = 0
        self.last_reward = 0.0
        self.last_demerit = 0.0
        self.total_demerits = 0.0
        self.fitness = 0.0
        self.done = False
        spawn_tile(self.board, self.rng)
        spawn_tile(self.board, self.rng)

    def valid_moves(self) -> list[str]:
        return [action for action in ACTIONS if move_board(self.board, action)[2]]

    def step(self, action: str) -> None:
        if self.done:
            return
        next_board, gained, changed = move_board(self.board, action)
        if not changed:
            self.last_demerit = 2.0
            self.total_demerits += self.last_demerit
            self.last_reward = -self.last_demerit
            self.fitness += self.last_reward
            self.done = not self.valid_moves()
            return

        self.board = next_board
        self.score += gained
        self.moves += 1
        spawn_tile(self.board, self.rng)
        self.last_demerit = 0.1 if gained == 0 else 0.0
        self.total_demerits += self.last_demerit
        self.last_reward = gained - self.last_demerit
        self.fitness += self.last_reward
        self.done = not self.valid_moves()

    def state(self) -> dict[str, Any]:
        return {
            "agentNumber": self.agent_id,
            "board": [row[:] for row in self.board],
            "score": self.score,
            "moves": self.moves,
            "lastReward": self.last_reward,
            "lastDemerit": self.last_demerit,
            "totalDemerits": self.total_demerits,
            "done": self.done,
            "highestTile": max(max(row) for row in self.board),
            "validMoves": self.valid_moves(),
        }


class LearningArena:
    def __init__(self, count: int = 15) -> None:
        self.lock = threading.Lock()
        self.running = True
        self.generation = 1
        self.tick = 0
        self.completed = 0
        self.high_score = 0
        self.global_weights = list(STARTING_WEIGHTS)
        self.history: list[dict[str, float | int]] = []
        self.count = 0
        self.games: list[PythonGame] = []
        self.weights: list[list[float]] = []
        self.set_count(count)

    def set_count(self, count: int) -> None:
        self.count = max(2, int(count))
        self.games = [PythonGame(index + 1, index + 1) for index in range(self.count)]
        self.weights = [self.mutate(self.global_weights, 0.6) for _ in self.games]
        self.completed = 0

    @staticmethod
    def mutate(weights: list[float], amount: float) -> list[float]:
        return [weight + (Random().random() * 2 - 1) * amount for weight in weights]

    def next_generation(self) -> None:
        ranked = sorted(zip(self.weights, self.games), key=lambda item: item[1].fitness, reverse=True)
        scores = [game.score for _, game in ranked]
        fitnesses = [game.fitness for _, game in ranked]
        demerits = [game.total_demerits for _, game in ranked]
        self.history.append({
            "generation": self.generation,
            "bestScore": max(scores),
            "averageScore": sum(scores) / len(scores),
            "bestFitness": max(fitnesses),
            "averageFitness": sum(fitnesses) / len(fitnesses),
            "averageDemerits": sum(demerits) / len(demerits),
        })
        total_weight = sum(max(game.fitness + 1, 0.1) for _, game in ranked)
        self.global_weights = [
            sum(weights[index] * max(game.fitness + 1, 0.1) for weights, game in ranked) / total_weight
            for index in range(len(FEATURES))
        ]
        elite_count = max(1, min(5, int(self.count * 0.3)))
        survivors = [weights for weights, _ in ranked[:elite_count]]
        mutation_amount = max(0.08, 0.35 * (1 - self.generation / 40))
        self.weights = [self.global_weights[:]] + survivors[: self.count - 1]
        while len(self.weights) < self.count:
            self.weights.append(breed(survivors, mutation_amount))
        self.generation += 1
        self.completed = 0
        for index, game in enumerate(self.games):
            game.reset(self.generation * 1000 + self.tick + index)

    def tick_once(self) -> None:
        with self.lock:
            if not self.running:
                return
            self.tick += 1
            for index, game in enumerate(self.games):
                if game.done:
                    continue
                action = policy(game.board, self.weights[index])
                game.step(action)
                self.high_score = max(self.high_score, game.score)
                if game.done:
                    self.completed += 1
            ordered = sorted(zip(self.games, self.weights), key=lambda item: item[0].done)
            self.games = [game for game, _ in ordered]
            self.weights = [weights for _, weights in ordered]
            if self.completed == self.count:
                self.next_generation()

    def state(self) -> dict[str, Any]:
        with self.lock:
            return {
                "games": [game.state() for game in self.games],
                "generation": self.generation,
                "highScore": self.high_score,
                "completed": self.completed,
                "count": self.count,
                "tick": self.tick,
                "running": self.running,
                "globalWeights": self.global_weights,
                "features": FEATURES,
                "history": self.history[-100:],
            }

    def command(self, payload: dict[str, Any]) -> None:
        with self.lock:
            command = payload.get("command")
            if command == "start":
                self.running = True
            elif command == "stop":
                self.running = False
            elif command == "reset":
                self.generation = 1
                self.tick = 0
                self.high_score = 0
                self.global_weights = list(STARTING_WEIGHTS)
                self.history = []
                self.set_count(self.count)
                self.running = False
            elif command == "count":
                self.global_weights = list(STARTING_WEIGHTS)
                self.generation = 1
                self.tick = 0
                self.high_score = 0
                self.history = []
                self.set_count(payload.get("count", self.count))
                self.running = True


arena = LearningArena()


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/state":
            self.send_json(200, arena.state())
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/api/command":
            self.send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        arena.command(payload)
        self.send_json(200, arena.state())

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def run() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Python learning server: http://{HOST}:{PORT}")
    try:
        while True:
            arena.tick_once()
            time.sleep(TICK_SECONDS)
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    run()
