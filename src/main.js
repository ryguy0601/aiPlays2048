import './style.css'

const PYTHON_API = 'http://127.0.0.1:8765'
const app = document.querySelector('#app')
let latestState = null
let pollTimer = null
let pendingCount = null

app.innerHTML = `
  <header class="topbar">
    <a class="brand" href="/" aria-label="AI plays 2048 home"><span>AI</span> PLAYS <strong>2048</strong></a>
    <div class="status"><span id="status-dot" class="status-dot"></span><span id="status-text">connecting to Python</span></div>
  </header>
  <main class="layout">
    <section class="arena-panel" aria-label="Python evolution arena">
      <div class="game-header">
        <div><span class="live-label">PYTHON ARENA</span><h2 id="run-label">connecting...</h2></div>
        <div class="arena-actions"><label class="population-setting">number of games <input id="game-count" aria-label="Number of games" type="number" min="2" value="15"></label><button id="apply-count" class="button button-light" type="button">Apply</button><button id="new-game" class="button button-light" type="button">↻ Reset</button><button id="toggle-agent" class="button button-accent" type="button">Stop agents</button></div>
      </div>
      <div class="arena-stats"><div><strong id="generation">-</strong><span>generation</span></div><div><strong id="high-score">0</strong><span>high score</span></div><div><strong id="completed">0 / 0</strong><span>runs complete</span></div></div>
      <div id="arena" class="arena" aria-live="polite"></div>
      <div class="arena-footer"><span>Python owns the games and model</span><span id="tick-label">tick 0</span></div>
      <div class="chart-grid"><div class="chart-card"><div class="chart-title"><strong>score progress</strong><span>best / average</span></div><canvas id="score-chart" height="180"></canvas></div><div class="chart-card"><div class="chart-title"><strong>learning quality</strong><span>fitness / demerits</span></div><canvas id="quality-chart" height="180"></canvas></div></div>
    </section>

    <aside class="agent-panel">
      <div class="panel-heading"><span class="panel-index">PY</span><h2>Python console</h2><span id="pulse" class="pulse"></span></div>
      <p class="panel-copy">The browser is a view. Python makes every decision, applies rewards and demerits, and evolves the global model.</p>
      <div class="console"><div><span class="prompt">&gt;</span> <span id="console-line">waiting for python/server.py...</span></div><div class="console-result" id="reward-line">global model: unavailable</div></div>
      <div class="agent-actions"><button id="demo-agent" class="button button-accent" type="button">Start Python agents</button><span class="hint">run python/server.py first</span></div>
      <div class="contract"><div><span>OWNER</span><strong>Python learning server</strong></div><div><span>MODEL</span><strong>global weighted policy</strong></div><div><span>REWARD</span><strong>score minus demerits</strong></div></div>
    </aside>
  </main>
  <footer><span>PYTHON CONTROLLED</span><span id="episode-label">0 parallel episodes</span><a href="readme.md">Python integration notes ↗</a></footer>
`

const arenaElement = document.querySelector('#arena')
const statusText = document.querySelector('#status-text')
const statusDot = document.querySelector('#status-dot')
const toggleButton = document.querySelector('#toggle-agent')
const demoButton = document.querySelector('#demo-agent')
const gameCountInput = document.querySelector('#game-count')
const applyCountButton = document.querySelector('#apply-count')
const consoleLine = document.querySelector('#console-line')
const rewardLine = document.querySelector('#reward-line')
const scoreChart = document.querySelector('#score-chart')
const qualityChart = document.querySelector('#quality-chart')

function boardMarkup(state) {
  const cells = state.board.flat().map((value) => `<div class="mini-cell tile-${value}" aria-label="${value || 'empty'}">${value || ''}</div>`).join('')
  return `<article class="agent-card ${state.done ? 'is-done' : ''}"><div class="agent-card-head"><span>AGENT ${String(state.agentNumber).padStart(2, '0')}</span><strong>${state.score}</strong></div><div class="mini-board">${cells}</div><div class="agent-card-foot"><span>${state.done ? 'waiting' : `${state.moves} moves`}</span><span>demerits ${state.totalDemerits.toFixed(1)}</span></div></article>`
}

function render(state) {
  latestState = state
  const activeCount = state.games.filter((game) => !game.done).length
  arenaElement.innerHTML = state.games.map(boardMarkup).join('')
  document.querySelector('#generation').textContent = state.generation
  document.querySelector('#high-score').textContent = state.highScore
  document.querySelector('#completed').textContent = `${state.completed} / ${state.count}`
  document.querySelector('#episode-label').textContent = `${state.count} parallel episodes`
  document.querySelector('#tick-label').textContent = `tick ${state.tick}`
  document.querySelector('#run-label').textContent = state.running ? `${activeCount} / ${state.count} agents active` : `${state.count} agents stopped`
  statusText.textContent = state.running ? 'Python running' : 'Python stopped'
  statusDot.classList.toggle('is-running', state.running)
  toggleButton.textContent = state.running ? 'Stop agents' : 'Start agents'
  demoButton.textContent = state.running ? 'Stop Python agents' : 'Start Python agents'
  const latest = state.games.reduce((best, game) => game.lastReward > best.lastReward ? game : best, state.games[0])
  rewardLine.textContent = `reward: ${latest.lastReward.toFixed(1)} / demerit: ${latest.lastDemerit.toFixed(1)}`
  consoleLine.textContent = `Python generation ${state.generation} / global model active`
  if (pendingCount === null && document.activeElement !== gameCountInput) gameCountInput.value = state.count
  drawChart(scoreChart, state.history, ["bestScore", "averageScore"], ['#e77b38', '#9bd1d0'], ['best score', 'average score'])
  drawChart(qualityChart, state.history, ["bestFitness", "averageDemerits"], ['#b5d544', '#e77b38'], ['best fitness', 'average demerits'])
}

function drawChart(canvas, history, keys, colors, labels) {
  const context = canvas.getContext('2d')
  const width = canvas.clientWidth || 320
  const height = canvas.clientHeight || 180
  const ratio = window.devicePixelRatio || 1
  canvas.width = width * ratio
  canvas.height = height * ratio
  context.setTransform(ratio, 0, 0, ratio, 0, 0)
  context.clearRect(0, 0, width, height)
  const padding = { top: 28, right: 10, bottom: 25, left: 42 }
  const plotWidth = width - padding.left - padding.right
  const plotHeight = height - padding.top - padding.bottom
  const maximum = Math.max(...keys.flatMap((key) => history?.map((entry) => Number(entry[key]) || 0) || []), 1)
  const tickStep = maximum / 4
  context.font = '9px DM Mono'
  context.textAlign = 'left'
  labels.forEach((label, index) => {
    context.fillStyle = colors[index]
    context.fillText(label, padding.left + index * 105, 12)
  })
  context.strokeStyle = '#d7d8cc'
  context.lineWidth = 1
  context.fillStyle = '#6a766f'
  context.textAlign = 'right'
  for (let row = 0; row <= 4; row += 1) {
    const y = padding.top + (plotHeight / 4) * row
    context.beginPath()
    context.moveTo(padding.left, y)
    context.lineTo(width - padding.right, y)
    context.stroke()
    context.fillText(Math.round(maximum - tickStep * row), padding.left - 7, y + 3)
  }
  if (!history?.length) {
    context.fillStyle = '#6a766f'
    context.font = '10px DM Mono'
    context.textAlign = 'left'
    context.fillText('waiting for generation 1 to finish...', padding.left, height / 2)
    return
  }
  context.fillStyle = '#6a766f'
  context.textAlign = 'center'
  const labelIndexes = [...new Set([0, Math.floor((history.length - 1) / 2), history.length - 1])]
  labelIndexes.forEach((index) => {
    const x = history.length === 1 ? padding.left + plotWidth / 2 : padding.left + (index / (history.length - 1)) * plotWidth
    context.fillText(`G${history[index].generation}`, x, height - 7)
  })
  keys.forEach((key, seriesIndex) => {
    context.strokeStyle = colors[seriesIndex]
    context.lineWidth = 2
    context.beginPath()
    history.forEach((entry, index) => {
      const x = history.length === 1 ? padding.left + plotWidth / 2 : padding.left + (index / (history.length - 1)) * plotWidth
      const y = padding.top + plotHeight - ((Number(entry[key]) || 0) / maximum) * plotHeight
      if (index === 0) context.moveTo(x, y)
      else context.lineTo(x, y)
    })
    context.stroke()
  })
}

async function getState() {
  const response = await fetch(`${PYTHON_API}/api/state`)
  if (!response.ok) throw new Error('Python server unavailable')
  return response.json()
}

async function sendCommand(command, count) {
  const response = await fetch(`${PYTHON_API}/api/command`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(count === undefined ? { command } : { command, count }),
  })
  if (!response.ok) throw new Error('Python command failed')
  const state = await response.json()
  render(state)
  if (command === 'count') {
    pendingCount = null
    gameCountInput.value = state.count
  }
}

async function poll() {
  try {
    render(await getState())
  } catch (error) {
    statusText.textContent = 'Python offline'
    statusDot.classList.remove('is-running')
    consoleLine.textContent = 'start python/server.py to connect'
    rewardLine.textContent = 'global model: unavailable'
  }
  pollTimer = setTimeout(poll, 120)
}

async function command(command, count) {
  try {
    await sendCommand(command, count)
  } catch {
    consoleLine.textContent = 'Python server is offline'
  }
}

toggleButton.addEventListener('click', () => command(latestState?.running ? 'stop' : 'start'))
demoButton.addEventListener('click', () => command(latestState?.running ? 'stop' : 'start'))
document.querySelector('#new-game').addEventListener('click', () => command('reset'))
applyCountButton.addEventListener('click', () => {
  const count = Math.max(2, Number.parseInt(gameCountInput.value, 10) || 15)
  pendingCount = count
  gameCountInput.value = count
  command('count', count)
})
gameCountInput.addEventListener('input', () => {
  pendingCount = gameCountInput.value
})
gameCountInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') applyCountButton.click()
})

window.ai2048 = {
  state: () => latestState,
  refresh: getState,
  command,
}

poll()
