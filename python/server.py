"""Python-owned 2048 learning server with dynamic ML Model Selection,
Hyperparameter Customization, and Portable JSON Export/Import.
"""

import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from random import Random, sample
from typing import Any

from evolve_agent import (
    ACTIONS,
    ML_MODELS,
    MLPModel,
    QLearningLinearModel,
    NTupleNetwork,
    breed_mlp,
    move_board,
    select_action,
    spawn_tile,
)

HOST = "127.0.0.1"
PORT = 8765
TICK_SECONDS = 0.075


class PythonGame:
    def __init__(self, agent_id: int, seed: int) -> None:
        self.agent_id = agent_id
        self.rng = Random(seed)
        self.board = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.moves = 0
        self.last_reward = 0.0
        self.last_demerit = 0.0
        self.total_demerits = 0.0
        self.fitness = 0.0
        self.done = False
        self.cached_highest = 0
        self.reset(seed)

    def reset(self, seed: int) -> None:
        self.rng.seed(seed)
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
        self.cached_highest = max(max(r) for r in self.board)

    def check_done(self) -> bool:
        for r in range(4):
            for c in range(4):
                if self.board[r][c] == 0:
                    return False
        for a in ACTIONS:
            if move_board(self.board, a)[2]:
                return False
        return True

    def step(self, action: str) -> None:
        if self.done:
            return

        next_board, gained, changed = move_board(self.board, action)
        if not changed:
            self.last_demerit = 2.0
            self.total_demerits += 2.0
            self.last_reward = -2.0
            self.fitness -= 2.0
            self.done = self.check_done()
            return

        self.board = next_board
        self.score += gained
        self.moves += 1
        spawn_tile(self.board, self.rng)

        self.last_demerit = 0.1 if gained == 0 else 0.0
        self.total_demerits += self.last_demerit
        self.last_reward = gained - self.last_demerit
        self.fitness += self.last_reward

        for r in range(4):
            for c in range(4):
                if self.board[r][c] > self.cached_highest:
                    self.cached_highest = self.board[r][c]

        self.done = self.check_done()

    def state(self) -> dict[str, Any]:
        return {
            "agentNumber": self.agent_id,
            "board": [r[:] for r in self.board],
            "score": self.score,
            "moves": self.moves,
            "lastReward": round(self.last_reward, 2),
            "lastDemerit": round(self.last_demerit, 2),
            "totalDemerits": round(self.total_demerits, 2),
            "done": self.done,
            "highestTile": self.cached_highest,
        }


class LearningArena:
    def __init__(self, count: int = 15) -> None:
        self.lock = threading.Lock()
        self.running = True
        self.generation = 1
        self.tick = 0
        self.completed = 0
        self.high_score = 0
        self.model_type = "mlp_neuroevolution"

        # Hyperparameters
        self.hyperparams = {
            "mutation_rate": 0.20,
            "mutation_strength": 0.30,
            "learning_rate": 0.02,
            "gamma": 0.95,
            "epsilon": 0.05,
        }

        self.history: list[dict[str, float | int]] = []
        self.count = 0
        self.games: list[PythonGame] = []
        self.models: list[Any] = []
        self.global_champion: Any = None
        self.set_model_type("mlp_neuroevolution", count=count)

    def set_model_type(self, m_type: str, params: dict | None = None, count: int | None = None) -> None:
        if m_type not in ML_MODELS:
            m_type = "mlp_neuroevolution"
        self.model_type = m_type
        if params:
            self.hyperparams.update(params)

        if count is not None:
            self.count = max(2, int(count))

        self.generation = 1
        self.tick = 0
        self.high_score = 0
        self.history.clear()
        self.games = [PythonGame(i + 1, i + 1) for i in range(self.count)]

        # Initialize paradigm
        if self.model_type == "mlp_neuroevolution":
            self.global_champion = MLPModel()
            self.models = [
                breed_mlp(self.global_champion, self.global_champion, self.hyperparams["mutation_rate"], self.hyperparams["mutation_strength"])
                for _ in range(self.count)
            ]
        elif self.model_type == "q_learning_linear":
            self.global_champion = QLearningLinearModel(
                lr=self.hyperparams["learning_rate"],
                gamma=self.hyperparams["gamma"],
                epsilon=self.hyperparams["epsilon"],
            )
            self.models = [self.global_champion.clone() for _ in range(self.count)]
        elif self.model_type == "n_tuple_network":
            self.global_champion = NTupleNetwork(lr=self.hyperparams["learning_rate"])
            self.models = [self.global_champion.clone() for _ in range(self.count)]

        self.completed = 0

    def next_generation(self) -> None:
        ranked = sorted(zip(self.models, self.games), key=lambda item: item[1].fitness, reverse=True)
        scores = [game.score for _, game in ranked]
        fitnesses = [game.fitness for _, game in ranked]
        demerits = [game.total_demerits for _, game in ranked]

        num_agents = len(scores)
        self.history.append({
            "generation": self.generation,
            "bestScore": max(scores),
            "averageScore": sum(scores) / num_agents,
            "bestFitness": max(fitnesses),
            "averageFitness": sum(fitnesses) / num_agents,
            "averageDemerits": sum(demerits) / num_agents,
        })

        self.global_champion = ranked[0][0].clone()

        if self.model_type == "mlp_neuroevolution":
            elite_count = max(2, min(6, int(self.count * 0.3)))
            survivors = [m for m, _ in ranked[:elite_count]]
            m_strength = max(0.04, self.hyperparams["mutation_strength"] * (1.0 - self.generation / 60.0))
            new_models = [self.global_champion.clone()]
            while len(new_models) < self.count:
                p1, p2 = sample(survivors, 2) if len(survivors) >= 2 else (survivors[0], survivors[0])
                new_models.append(breed_mlp(p1, p2, self.hyperparams["mutation_rate"], m_strength))
            self.models = new_models
        else:
            # Synchronize models toward the champion
            self.models = [self.global_champion.clone() for _ in range(self.count)]

        self.generation += 1
        self.completed = 0
        seed_base = self.generation * 1000 + self.tick
        for idx, game in enumerate(self.games):
            game.reset(seed_base + idx)

    def tick_once(self) -> None:
        with self.lock:
            if not self.running:
                return

            self.tick += 1
            for index, game in enumerate(self.games):
                if game.done:
                    continue

                model = self.models[index]
                action, next_val = select_action(game.board, model)

                prior_board = [r[:] for r in game.board]
                game.step(action)

                if not game.done:
                    if self.model_type == "q_learning_linear":
                        phi = model.extract_features(prior_board)
                        model.update_td(phi, game.last_reward, next_val)
                    elif self.model_type == "n_tuple_network":
                        model.update_td(prior_board, game.last_reward, next_val)

                if game.score > self.high_score:
                    self.high_score = game.score
                if game.done:
                    self.completed += 1

            if self.completed >= self.count:
                self.next_generation()
            else:
                ordered = sorted(zip(self.games, self.models), key=lambda item: item[0].done)
                self.games = [g for g, _ in ordered]
                self.models = [m for _, m in ordered]

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "games": [g.state() for g in self.games],
                "generation": self.generation,
                "highScore": self.high_score,
                "completed": self.completed,
                "count": self.count,
                "tick": self.tick,
                "running": self.running,
                "modelType": self.model_type,
                "hyperparameters": self.hyperparams,
                "availableModels": ML_MODELS,
                "history": self.history[-100:],
            }

    def export_model_json(self) -> dict[str, Any]:
        with self.lock:
            return {
                "model_type": self.model_type,
                "generation": self.generation,
                "high_score": self.high_score,
                "hyperparameters": dict(self.hyperparams),
                "model_data": self.global_champion.to_dict(),
            }

    def import_model_json(self, payload: dict[str, Any]) -> None:
        with self.lock:
            m_type = payload.get("model_type", "mlp_neuroevolution")
            m_data = payload.get("model_data", {})
            self.model_type = m_type
            if "hyperparameters" in payload:
                self.hyperparams.update(payload["hyperparameters"])

            if m_type == "mlp_neuroevolution":
                self.global_champion = MLPModel.from_dict(m_data)
            elif m_type == "q_learning_linear":
                self.global_champion = QLearningLinearModel.from_dict(m_data)
            elif m_type == "n_tuple_network":
                self.global_champion = NTupleNetwork.from_dict(m_data)
            else:
                raise ValueError(f"Unknown model type: {m_type}")

            self.generation = payload.get("generation", 1)
            self.high_score = max(self.high_score, payload.get("high_score", 0))
            self.models = [self.global_champion.clone() for _ in range(self.count)]
            self.completed = 0
            for i, game in enumerate(self.games):
                game.reset(self.generation * 1000 + i)

    def command(self, payload: dict[str, Any]) -> None:
        with self.lock:
            cmd = payload.get("command")
            if cmd == "start":
                self.running = True
            elif cmd == "stop":
                self.running = False
            elif cmd in ("reset", "restart"):
                self.set_model_type(self.model_type, count=payload.get("count", self.count))
                self.running = payload.get("running", True)
            elif cmd == "configure_model":
                m_type = payload.get("modelType", self.model_type)
                params = payload.get("hyperparameters", {})
                count = payload.get("count", self.count)
                self.set_model_type(m_type, params=params, count=count)
                self.running = True


arena = LearningArena()


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, data: dict[str, Any]) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/state":
            self.send_json(200, arena.snapshot())
        elif self.path == "/api/model/export":
            try:
                export_dict = arena.export_model_json()
                self.send_json(200, export_dict)
            except Exception as e:
                traceback.print_exc()
                self.send_json(500, {"error": f"Failed to export model: {str(e)}"})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw)
        except Exception as e:
            self.send_json(400, {"error": f"Malformed JSON: {str(e)}"})
            return

        if self.path == "/api/command":
            try:
                arena.command(payload)
                self.send_json(200, arena.snapshot())
            except Exception as e:
                traceback.print_exc()
                self.send_json(500, {"error": f"Command execution failed: {str(e)}"})
        elif self.path == "/api/model/import":
            try:
                arena.import_model_json(payload)
                self.send_json(200, {"status": "ok", "generation": arena.generation, "modelType": arena.model_type})
            except Exception as e:
                traceback.print_exc()
                self.send_json(500, {"error": f"Failed to import model: {str(e)}"})
        else:
            self.send_json(404, {"error": "not found"})

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def run() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Multi-Paradigm 2048 Server running on http://{HOST}:{PORT}")
    try:
        while True:
            arena.tick_once()
            time.sleep(TICK_SECONDS)
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    run()