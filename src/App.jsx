import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import './custom-bootstrap.scss';

const PYTHON_API = 'http://127.0.0.1:8765';

// ==========================================
// 1. ATOMIC PRESENTATIONAL COMPONENTS
// ==========================================

const Tile = React.memo(({ value }) => (
  <div className={`tile tile-${value || 0}`} aria-label={value ? `Tile ${value}` : 'Empty'}>
    {value > 0 ? value : ''}
  </div>
));

const MiniBoard = React.memo(({ board }) => (
  <div className="mini-board-grid">
    {board.map((row, rIdx) =>
      row.map((val, cIdx) => (
        <Tile key={`${rIdx}-${cIdx}`} value={val} />
      ))
    )}
  </div>
));

// ==========================================
// 2. CANVAS TELEMETRY COMPONENT
// ==========================================

const TelemetryChart = React.memo(({ history, keys, colors, labels, height = 130 }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !history || history.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.clientWidth || 280;
    const h = canvas.clientHeight || height;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = width * ratio;
    canvas.height = h * ratio;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, h);

    const pad = { top: 22, right: 12, bottom: 20, left: 38 };
    const pW = width - pad.left - pad.right;
    const pH = h - pad.top - pad.bottom;

    let maxVal = 1;
    for (let i = 0; i < history.length; i++) {
      for (let k = 0; k < keys.length; k++) {
        const val = Number(history[i][keys[k]]) || 0;
        if (val > maxVal) maxVal = val;
      }
    }

    ctx.font = '9px "DM Mono", monospace';
    ctx.textAlign = 'left';
    labels.forEach((label, i) => {
      ctx.fillStyle = colors[i];
      ctx.fillText(label.toUpperCase(), pad.left + i * 105, 12);
    });

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    for (let r = 0; r <= 4; r++) {
      const y = pad.top + pH * (1 - r / 4);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(width - pad.right, y);
      ctx.stroke();
    }

    const len = Math.max(history.length - 1, 1);
    keys.forEach((key, kIdx) => {
      ctx.strokeStyle = colors[kIdx];
      ctx.lineWidth = 1.75;
      ctx.beginPath();
      let first = true;
      for (let idx = 0; idx < history.length; idx++) {
        const val = Number(history[idx][key]) || 0;
        const x = pad.left + (idx / len) * pW;
        const y = pad.top + pH * (1 - val / maxVal);
        if (first) {
          ctx.moveTo(x, y);
          first = false;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    });
  }, [history, keys, colors, labels, height]);

  return <canvas ref={canvasRef} style={{ width: '100%', height: `${height}px`, display: 'block' }} />;
});

// ==========================================
// 3. AGENT CARD & ML CONFIG PANEL
// ==========================================

const AgentGridCard = React.memo(
  ({ game, isLeader, onSelect }) => {
    return (
      <div className="col">
        <div
          className={`card arena-agent-card h-100 ${game.done ? 'is-eliminated' : ''} ${isLeader ? 'is-leader' : ''}`}
          onClick={() => onSelect(game)}
        >
          <div className="card-body p-2 d-flex flex-column gap-1">
            <div className="d-flex justify-content-between align-items-center">
              <span className="badge agent-tag">
                {isLeader && <span className="leader-star me-1">★</span>}
                #{String(game.agentNumber).padStart(2, '0')}
              </span>
              <strong className="text-white small font-monospace">{game.score}</strong>
            </div>
            <div className="agent-mini-board my-1">
              <MiniBoard board={game.board} />
            </div>
            <div className="d-flex justify-content-between agent-stats-strip">
              <span className={game.done ? 'text-danger' : 'text-muted'}>
                {game.done ? 'DONE' : `${game.moves || 0} MOVES`}
              </span>
              <span className="text-muted">D: {Number(game.totalDemerits || 0).toFixed(1)}</span>
            </div>
          </div>
        </div>
      </div>
    );
  },
  (prev, next) => (
    prev.isLeader === next.isLeader &&
    prev.game.moves === next.game.moves &&
    prev.game.score === next.game.score &&
    prev.game.done === next.game.done &&
    prev.game.highestTile === next.game.highestTile
  )
);

const MLModelConfigPanel = React.memo(({ currentModel, hyperparams, onApplyConfig }) => {
  const [selectedModel, setSelectedModel] = useState(currentModel || 'mlp_neuroevolution');
  const [params, setParams] = useState(hyperparams || {});
  const [isDirty, setIsDirty] = useState(false);

  useEffect(() => {
    if (!isDirty) {
      if (currentModel) setSelectedModel(currentModel);
      if (hyperparams) setParams(hyperparams);
    }
  }, [currentModel, hyperparams, isDirty]);

  const handleModelChange = (e) => {
    setSelectedModel(e.target.value);
    setIsDirty(true);
  };

  const handleParamChange = (k, rawValue) => {
    setIsDirty(true);
    setParams((prev) => ({ ...prev, [k]: rawValue }));
  };

  const submit = () => {
    const cleanedParams = {};
    Object.keys(params).forEach((k) => {
      const parsed = parseFloat(params[k]);
      cleanedParams[k] = isNaN(parsed) ? (hyperparams?.[k] ?? 0.0) : parsed;
    });

    onApplyConfig(selectedModel, cleanedParams);
    setIsDirty(false);
  };

  return (
    <div className="card mb-3 shadow-sm">
      <div className="card-header d-flex justify-content-between align-items-center py-2 px-3 border-bottom">
        <span className="fw-bold small text-uppercase">Learning Paradigm</span>
        <span className={`badge ${isDirty ? 'bg-primary' : 'bg-secondary'}`}>
          {isDirty ? 'MODIFIED' : 'SYNCED'}
        </span>
      </div>
      <div className="card-body p-3 d-flex flex-column gap-3">
        <div>
          <label className="form-label text-muted small text-uppercase mb-1">Architecture</label>
          <select
            className="form-select form-select-sm"
            value={selectedModel}
            onChange={handleModelChange}
          >
            <option value="mlp_neuroevolution">Deep Neuroevolution (MLP)</option>
            <option value="q_learning_linear">TD Q-Learning (Linear)</option>
            <option value="n_tuple_network">N-Tuple Network (TD Patterns)</option>
          </select>
        </div>

        <div className="bg-black p-2 rounded border border-dark">
          {selectedModel === 'mlp_neuroevolution' && (
            <div className="d-flex flex-column gap-2">
              <div className="d-flex justify-content-between align-items-center">
                <span className="small text-muted">Mutation Rate</span>
                <input
                  type="number"
                  step="0.05"
                  min="0.01"
                  max="1.0"
                  className="form-control form-control-sm text-end w-auto"
                  value={params.mutation_rate ?? 0.20}
                  onChange={(e) => handleParamChange('mutation_rate', e.target.value)}
                />
              </div>
              <div className="d-flex justify-content-between align-items-center">
                <span className="small text-muted">Mutation Strength</span>
                <input
                  type="number"
                  step="0.05"
                  min="0.01"
                  max="2.0"
                  className="form-control form-control-sm text-end w-auto"
                  value={params.mutation_strength ?? 0.30}
                  onChange={(e) => handleParamChange('mutation_strength', e.target.value)}
                />
              </div>
            </div>
          )}

          {selectedModel === 'q_learning_linear' && (
            <div className="d-flex flex-column gap-2">
              <div className="d-flex justify-content-between align-items-center">
                <span className="small text-muted">Learning Rate (α)</span>
                <input
                  type="number"
                  step="0.005"
                  min="0.001"
                  max="0.5"
                  className="form-control form-control-sm text-end w-auto"
                  value={params.learning_rate ?? 0.02}
                  onChange={(e) => handleParamChange('learning_rate', e.target.value)}
                />
              </div>
              <div className="d-flex justify-content-between align-items-center">
                <span className="small text-muted">Discount (γ)</span>
                <input
                  type="number"
                  step="0.01"
                  min="0.5"
                  max="0.99"
                  className="form-control form-control-sm text-end w-auto"
                  value={params.gamma ?? 0.95}
                  onChange={(e) => handleParamChange('gamma', e.target.value)}
                />
              </div>
              <div className="d-flex justify-content-between align-items-center">
                <span className="small text-muted">Exploration (ε)</span>
                <input
                  type="number"
                  step="0.01"
                  min="0.0"
                  max="0.5"
                  className="form-control form-control-sm text-end w-auto"
                  value={params.epsilon ?? 0.05}
                  onChange={(e) => handleParamChange('epsilon', e.target.value)}
                />
              </div>
            </div>
          )}

          {selectedModel === 'n_tuple_network' && (
            <div className="d-flex justify-content-between align-items-center">
              <span className="small text-muted">Tuple Rate (α)</span>
              <input
                type="number"
                step="0.005"
                min="0.001"
                max="0.2"
                className="form-control form-control-sm text-end w-auto"
                value={params.learning_rate ?? 0.02}
                onChange={(e) => handleParamChange('learning_rate', e.target.value)}
              />
            </div>
          )}
        </div>

        <button className="btn btn-sm btn-info w-100 fw-bold text-dark" onClick={submit}>
          DEPLOY CONFIG
        </button>
      </div>
    </div>
  );
});

const LeaderSpotlight = React.memo(({ leader }) => (
  <div className="card shadow-sm">
    <div className="card-header d-flex justify-content-between align-items-center py-2 px-3 border-bottom">
      <span className="fw-bold small text-uppercase">Leader Spotlight</span>
      <span className="badge bg-primary">BEST EVAL</span>
    </div>
    <div className="card-body p-3">
      {leader ? (
        <div className="d-flex flex-column gap-2">
          <div className="d-flex justify-content-between small p-2 bg-black rounded border border-dark">
            <div><span className="text-muted">AGENT:</span> <strong className="text-info">#{leader.agentNumber}</strong></div>
            <div><span className="text-muted">SCORE:</span> <strong>{leader.score}</strong></div>
            <div><span className="text-muted">MAX:</span> <strong className="text-primary">{leader.highestTile}</strong></div>
          </div>
          <div className="mx-auto w-100" style={{ maxWidth: '170px' }}>
            <div className="leader-board border border-primary p-1 rounded">
              <MiniBoard board={leader.board} />
            </div>
          </div>
        </div>
      ) : (
        <div className="text-muted small text-center py-4">Awaiting evaluations...</div>
      )}
    </div>
  </div>
));

// ==========================================
// 4. MAIN APP COMPONENT
// ==========================================

const App = () => {
  const [latestState, setLatestState] = useState(null);
  const [connectionError, setConnectionError] = useState(false);
  const [gameCount, setGameCount] = useState(15);
  const [inspectedAgent, setInspectedAgent] = useState(null);
  const [ioStatus, setIoStatus] = useState('');
  const fileInputRef = useRef(null);

  const pollState = useCallback(async () => {
    try {
      const res = await fetch(`${PYTHON_API}/api/state`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setLatestState({ ...data, isConnected: true });
      setConnectionError(false);
    } catch (err) {
      setConnectionError(true);
      setLatestState((prev) => (prev ? { ...prev, isConnected: false } : null));
    }
  }, []);

  useEffect(() => {
    pollState();
    const interval = setInterval(pollState, 100);
    return () => clearInterval(interval);
  }, [pollState]);

  const sendCommand = async (payload) => {
    try {
      await fetch(`${PYTHON_API}/api/command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      pollState();
    } catch (e) {
      console.error('Command failed:', e.message);
    }
  };

  const handleConfigureModel = (modelType, hyperparameters) => {
    sendCommand({
      command: 'configure_model',
      modelType,
      hyperparameters,
      count: parseInt(gameCount, 10),
    });
  };

  const handleExportModel = async () => {
    try {
      setIoStatus('Exporting...');
      const res = await fetch(`${PYTHON_API}/api/model/export`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
      });

      if (!res.ok) {
        let errMessage = `HTTP ${res.status}`;
        try {
          const errData = await res.json();
          if (errData?.error) errMessage = errData.error;
        } catch (_) {}
        throw new Error(errMessage);
      }

      const modelJson = await res.json();
      const jsonString = JSON.stringify(modelJson, null, 2);
      const blob = new Blob([jsonString], { type: 'application/json;charset=utf-8;' });
      const downloadUrl = window.URL.createObjectURL(blob);

      const filename = `2048_${modelJson.model_type || 'model'}_gen${modelJson.generation || 1}_score${modelJson.high_score || 0}.json`;

      const link = document.createElement('a');
      link.href = downloadUrl;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();

      setTimeout(() => {
        document.body.removeChild(link);
        window.URL.revokeObjectURL(downloadUrl);
      }, 150);

      setIoStatus('Exported!');
      setTimeout(() => setIoStatus(''), 3000);
    } catch (err) {
      console.error('Export Error:', err);
      setIoStatus('Export Failed');
      alert(`Export Failed: ${err.message}`);
      setTimeout(() => setIoStatus(''), 3000);
    }
  };

  const handleImportFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setIoStatus('Uploading...');
      const text = await file.text();
      const parsed = JSON.parse(text);

      const res = await fetch(`${PYTHON_API}/api/model/import`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(parsed),
      });

      if (!res.ok) {
        let errMessage = `HTTP ${res.status}`;
        try {
          const errData = await res.json();
          if (errData?.error) errMessage = errData.error;
        } catch (_) {}
        throw new Error(errMessage);
      }

      setIoStatus('Model Loaded!');
      pollState();
      setTimeout(() => setIoStatus(''), 3000);
    } catch (err) {
      alert(`Failed to load model: ${err.message}`);
      setIoStatus('Load Error');
      setTimeout(() => setIoStatus(''), 3000);
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleToggleRunning = () => sendCommand({ command: latestState?.running ? 'stop' : 'start' });
  const handleRestart = () => sendCommand({ command: 'reset', count: parseInt(gameCount, 10), running: true });

  const games = latestState?.games || [];

  const leaderAgent = useMemo(() => {
    if (!games.length) return null;
    return games.reduce((max, g) => (g.score > (max?.score || 0) ? g : max), games[0]);
  }, [games]);

  const activeCount = useMemo(() => games.filter((g) => !g.done).length, [games]);

  if (!latestState && !connectionError) {
    return (
      <div className="d-flex flex-column align-items-center justify-content-center vh-100 gap-3">
        <div className="spinner-border text-primary" role="status"></div>
        <div className="small font-monospace text-primary">INITIALIZING MACHINE LEARNING ARENA...</div>
      </div>
    );
  }

  if (!latestState && connectionError) {
    return (
      <div className="d-flex flex-column align-items-center justify-content-center vh-100 gap-2">
        <span className="badge bg-danger">OFFLINE</span>
        <h3 className="text-white mt-2">NEURAL BACKEND OFFLINE</h3>
        <p className="text-muted small">Ensure <code>python server.py</code> is active on <code>{PYTHON_API}</code></p>
        <button className="btn btn-sm btn-info text-dark mt-2" onClick={pollState}>RECONNECT</button>
      </div>
    );
  }

  const isRunning = latestState?.running;

  return (
    <div className="container-fluid p-0 d-flex flex-column min-vh-100">
      <input
        ref={fileInputRef}
        type="file"
        accept=".json"
        className="d-none"
        onChange={handleImportFile}
      />

      {/* 1. TOP NAVBAR / HUD */}
      <nav className="navbar navbar-expand-xl px-3 py-2 sticky-top hud-bar">
        <div className="container-fluid px-0 gap-2">
          <div className="d-flex align-items-center gap-2">
            <span className="badge bg-primary px-2 py-1 fs-6">RL-2048</span>
            <span className="navbar-brand text-white fw-bold mb-0 fs-6 font-syne">COCKPIT</span>
          </div>

          <div className="d-flex flex-wrap align-items-center gap-3 bg-black py-1 px-3 rounded border border-dark">
            <div className="d-flex flex-column">
              <span className="hud-metric-lbl">ACTIVE MODEL</span>
              <strong className="text-primary small">
                {latestState.modelType === 'mlp_neuroevolution' && 'DEEP MLP'}
                {latestState.modelType === 'q_learning_linear' && 'Q-LEARNING'}
                {latestState.modelType === 'n_tuple_network' && 'N-TUPLE'}
              </strong>
            </div>
            <div className="vr bg-secondary opacity-50"></div>
            <div className="d-flex flex-column">
              <span className="hud-metric-lbl">GEN</span>
              <strong className="text-info small">{latestState.generation ?? 1}</strong>
            </div>
            <div className="vr bg-secondary opacity-50"></div>
            <div className="d-flex flex-column">
              <span className="hud-metric-lbl">HIGH SCORE</span>
              <strong className="text-white small">{latestState.highScore ?? 0}</strong>
            </div>
            <div className="vr bg-secondary opacity-50"></div>
            <div className="d-flex flex-column">
              <span className="hud-metric-lbl">ACTIVE / POOL</span>
              <strong className="text-muted small">{activeCount} / {latestState.count}</strong>
            </div>
          </div>

          <div className="d-flex flex-wrap align-items-center gap-2 ms-auto">
            <div className="input-group input-group-sm w-auto">
              <span className="input-group-text bg-dark border-secondary text-muted small">POOLS</span>
              <input
                type="number"
                min="2"
                max="50"
                className="form-control bg-black text-info border-secondary text-center"
                style={{ width: '60px' }}
                value={gameCount}
                onChange={(e) => setGameCount(e.target.value)}
              />
            </div>

            <div className="btn-group btn-group-sm">
              <button className="btn btn-outline-info" onClick={handleExportModel} title="Export model state as JSON">
                ↓ EXPORT
              </button>
              <button className="btn btn-outline-info" onClick={() => fileInputRef.current?.click()} title="Upload model JSON from PC">
                ↑ UPLOAD
              </button>
            </div>
            {ioStatus && <span className="small text-primary fw-bold ms-1">{ioStatus}</span>}

            <button className="btn btn-sm btn-outline-danger" onClick={handleRestart} title="Reset learning progress">
              ↻ RESET
            </button>

            <button
              className={`btn btn-sm fw-bold ${isRunning ? 'btn-primary' : 'btn-info text-dark'}`}
              onClick={handleToggleRunning}
            >
              {isRunning ? '■ HALT' : '▶ TRAIN'}
            </button>
          </div>
        </div>
      </nav>

      {/* 2. THREE-PANEL COCKPIT GRID */}
      <div className="container-fluid flex-grow-1 p-3">
        <div className="row g-3 h-100">
          
          {/* Left Column: Algorithm Selector & Hyperparameters */}
          <div className="col-12 col-xl-3 col-lg-4 d-flex flex-column">
            <MLModelConfigPanel
              currentModel={latestState.modelType}
              hyperparams={latestState.hyperparameters}
              onApplyConfig={handleConfigureModel}
            />
            <LeaderSpotlight leader={leaderAgent} />
          </div>

          {/* Center Column: Game Grid */}
          <div className="col-12 col-xl-6 col-lg-8 d-flex flex-column">
            <div className="card flex-grow-1 shadow-sm">
              <div className="card-header d-flex justify-content-between align-items-center py-2 px-3 border-bottom">
                <span className="fw-bold small text-uppercase text-primary">PARALLEL POPULATION ({games.length} AGENTS)</span>
                <span className="d-flex align-items-center gap-2 small text-muted">
                  <span className={`status-orb ${isRunning ? 'live' : 'idle'}`}></span>
                  {isRunning ? 'INFERENCE ACTIVE' : 'IDLE'}
                </span>
              </div>
              <div className="card-body p-2 overflow-auto" style={{ maxHeight: 'calc(100vh - 180px)' }}>
                <div className="row row-cols-2 row-cols-sm-3 row-cols-md-4 row-cols-lg-3 row-cols-xxl-4 g-2">
                  {games.map((g) => (
                    <AgentGridCard
                      key={g.agentNumber}
                      game={g}
                      isLeader={leaderAgent?.agentNumber === g.agentNumber}
                      onSelect={setInspectedAgent}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Telemetry & Log */}
          <div className="col-12 col-xl-3 col-lg-12 d-flex flex-column gap-3">
            <div className="card shadow-sm">
              <div className="card-header d-flex justify-content-between align-items-center py-2 px-3 border-bottom">
                <span className="fw-bold small text-uppercase">SCORE PROGRESSION</span>
                <span className="badge bg-secondary">BEST / MEAN</span>
              </div>
              <div className="card-body p-2">
                <TelemetryChart
                  history={latestState.history}
                  keys={['bestScore', 'averageScore']}
                  colors={['#ff007f', '#a855f7']}
                  labels={['best score', 'avg score']}
                  height={130}
                />
              </div>
            </div>

            <div className="card shadow-sm">
              <div className="card-header d-flex justify-content-between align-items-center py-2 px-3 border-bottom">
                <span className="fw-bold small text-uppercase">FITNESS DYNAMICS</span>
                <span className="badge bg-secondary">OPTIMIZATION</span>
              </div>
              <div className="card-body p-2">
                <TelemetryChart
                  history={latestState.history}
                  keys={['bestFitness', 'averageDemerits']}
                  colors={['#00f0ff', '#ff007f']}
                  labels={['best fitness', 'demerits']}
                  height={130}
                />
              </div>
            </div>

            <div className="card flex-grow-1 shadow-sm">
              <div className="card-header py-2 px-3 border-bottom">
                <span className="fw-bold small text-uppercase">TRAINING LOG</span>
              </div>
              <div className="card-body p-2 font-monospace small bg-black rounded m-2 border border-dark text-muted" style={{ maxHeight: '110px', overflowY: 'auto' }}>
                <div>&gt; Model Paradigm: {latestState.modelType}</div>
                <div>&gt; Epoch/Generation: #{latestState.generation} | Pools: {latestState.count}</div>
                <div>&gt; Engine State: {isRunning ? 'Running @ 75ms' : 'Halted'}</div>
                {isRunning && <div>&gt; Tick #{latestState.tick} cycle advancing...</div>}
              </div>
            </div>

          </div>
        </div>
      </div>

      {/* 3. INSPECTION MODAL */}
      {inspectedAgent && (
        <div className="modal fade show d-block" tabIndex="-1" role="dialog" style={{ backgroundColor: 'rgba(0,0,0,0.75)' }}>
          <div className="modal-dialog modal-dialog-centered" role="document">
            <div className="modal-content shadow">
              <div className="modal-header py-2 px-3 border-bottom">
                <h6 className="modal-title font-monospace text-white">
                  AGENT #{String(inspectedAgent.agentNumber).padStart(2, '0')} INSPECTOR
                </h6>
                <button type="button" className="btn-close btn-close-white" onClick={() => setInspectedAgent(null)}></button>
              </div>
              <div className="modal-body p-3">
                <div className="row align-items-center">
                  <div className="col-6">
                    <div className="leader-board p-1 border border-secondary rounded">
                      <MiniBoard board={inspectedAgent.board} />
                    </div>
                  </div>
                  <div className="col-6 font-monospace small d-flex flex-column gap-2">
                    <div className="d-flex justify-content-between border-bottom border-dark pb-1">
                      <span className="text-muted">STATUS:</span>
                      <strong className={inspectedAgent.done ? 'text-danger' : 'text-success'}>
                        {inspectedAgent.done ? 'DONE' : 'ACTIVE'}
                      </strong>
                    </div>
                    <div className="d-flex justify-content-between border-bottom border-dark pb-1">
                      <span className="text-muted">SCORE:</span>
                      <strong>{inspectedAgent.score}</strong>
                    </div>
                    <div className="d-flex justify-content-between border-bottom border-dark pb-1">
                      <span className="text-muted">MOVES:</span>
                      <strong>{inspectedAgent.moves}</strong>
                    </div>
                    <div className="d-flex justify-content-between border-bottom border-dark pb-1">
                      <span className="text-muted">MAX TILE:</span>
                      <strong className="text-primary">{inspectedAgent.highestTile}</strong>
                    </div>
                    <div className="d-flex justify-content-between border-bottom border-dark pb-1">
                      <span className="text-muted">LAST REWARD:</span>
                      <strong>{inspectedAgent.lastReward}</strong>
                    </div>
                    <div className="d-flex justify-content-between">
                      <span className="text-muted">DEMERITS:</span>
                      <strong>{inspectedAgent.totalDemerits}</strong>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;