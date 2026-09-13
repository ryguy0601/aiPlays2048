# AI plays 2048

A browser-based 2048 arena where simple agents evolve their strategies through
population-based learning. The browser renders the games and charts; Python
owns the game state, rewards, policies, generations, and evolutionary updates.

## Features

- Run a configurable population of 2048 agents in parallel.
- Evolve six board features from a shared global model.
- Breed, mutate, and score policies across generations.
- Track best and average scores, fitness, and demerits in the browser.
- Keep dead boards visible until the entire cohort completes an episode.

## Requirements

- Node.js 18 or newer
- npm
- Python 3.10 or newer
- A Python environment available through `conda` (the default npm script uses
	the `base` environment)

## Getting started

Install the JavaScript dependencies:

```bash
npm install
```

Start the Python learning server in one terminal:

```bash
npm run python
```

Start the Vite development server in a second terminal:

```bash
npm run dev
```

Open the local URL printed by Vite. The browser connects to the Python server
at `http://127.0.0.1:8765`. Set the population size in the arena header and
select **Apply**. The population must contain at least two games.

To create a production build:

```bash
npm run build
```

## How the learning works

For each candidate policy, the server:

1. Resets a seeded board.
2. Uses the policy to choose actions from the board state.
3. Adds rewards and demerits to the policy's fitness.
4. Keeps the strongest policies, crosses them over, and mutates the rest.

The server starts with six zero weights and evolves these board features:

- Empty spaces
- Largest tile
- Largest-tile corner placement
- Smooth neighboring values
- Merge opportunities
- Monotonic rows or columns

Rewards are shaped to discourage wasted actions. A merge returns its points, a
valid move without a merge receives a `-0.1` demerit, and an invalid move
receives a `-2` demerit. The arena uses these values to rank policies.

## Project structure

```text
src/                 Browser UI and styles
python/evolve_agent.py  2048 rules and policy evolution
python/server.py      Python game server and API
index.html            Browser entry point
package.json          npm scripts and dependencies
```

## Development notes

The Python server keeps chart history in memory for the current session. A
server restart resets the history, generation, and population. The frontend is
implemented with Vite and vanilla JavaScript; no frontend framework is
required.
