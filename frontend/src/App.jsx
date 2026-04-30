import { useState, useEffect, useRef, useCallback } from "react";
import { io } from "socket.io-client";
import "./App.css";

const API = "http://localhost:5000";

function App() {
  const [events, setEvents] = useState([]);
  const [alerts, setAlerts] = useState({});
  const [stats, setStats] = useState({
    frame_num: 0, fps: 0, total_tracked: 0,
    active_alerts: 0, progress: 0, status: "idle",
  });
  const [config, setConfig] = useState({
    presence_threshold: 10, loitering_threshold: 30,
  });
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState("");
  const [connected, setConnected] = useState(false);
  const [showPhoneInput, setShowPhoneInput] = useState(false);
  const [phoneUrl, setPhoneUrl] = useState("");

  // Zone state
  const [zonePoints, setZonePoints] = useState([]);
  const [drawingZone, setDrawingZone] = useState(false);
  const [zoneSaved, setZoneSaved] = useState(false);

  const logRef = useRef(null);
  const socketRef = useRef(null);
  const debounceRef = useRef({});
  const videoContainerRef = useRef(null);

  // ── SocketIO connection ──
  useEffect(() => {
    const socket = io(API, { transports: ["websocket", "polling"] });
    socketRef.current = socket;

    socket.on("connect", () => setConnected(true));
    socket.on("disconnect", () => setConnected(false));

    socket.on("event", (evt) => {
      setEvents((prev) => [...prev, evt]);
    });

    socket.on("alert", (alert) => {
      setAlerts((prev) => ({ ...prev, [alert.id]: alert }));
    });

    socket.on("alert_clear", ({ id }) => {
      setAlerts((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
    });

    socket.on("stats", (s) => setStats(s));

    return () => socket.disconnect();
  }, []);

  // ── Auto-scroll event log ──
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [events]);

  // ── Fetch videos + zone on mount ──
  useEffect(() => {
    fetch(`${API}/api/videos`)
      .then((r) => r.json())
      .then(setVideos)
      .catch(() => {});

    fetch(`${API}/api/config`)
      .then((r) => r.json())
      .then(setConfig)
      .catch(() => {});

    fetch(`${API}/api/zone`)
      .then((r) => r.json())
      .then((data) => {
        if (data.points && data.points.length >= 3) {
          setZonePoints(data.points);
          setZoneSaved(true);
        }
      })
      .catch(() => {});
  }, []);

  // ── Polling fallback for alerts & stats ──
  useEffect(() => {
    const interval = setInterval(() => {
      fetch(`${API}/api/alerts`)
        .then((r) => r.json())
        .then((alertList) => {
          const alertMap = {};
          alertList.forEach((a) => { alertMap[a.id] = a; });
          setAlerts(alertMap);
        })
        .catch(() => {});

      fetch(`${API}/api/stats`)
        .then((r) => r.json())
        .then(setStats)
        .catch(() => {});
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // ── Load first frame on video selection ──
  useEffect(() => {
    if (selectedVideo) {
      fetch(`${API}/api/load_video`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_path: selectedVideo }),
      }).catch(() => {});
    }
  }, [selectedVideo]);

  // ── Handlers ──
  const handleStart = () => {
    if (!selectedVideo) return;
    setEvents([]);
    setAlerts({});
    fetch(`${API}/api/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_path: selectedVideo }),
    });
  };

  const handleStop = () => {
    fetch(`${API}/api/stop`, { method: "POST" });
  };

  const handleConfigChange = (key, value) => {
    const updated = { ...config, [key]: value };
    setConfig(updated);

    if (debounceRef.current[key]) clearTimeout(debounceRef.current[key]);
    debounceRef.current[key] = setTimeout(() => {
      fetch(`${API}/api/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ [key]: value }),
      });
    }, 300);
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    // Clear input to allow uploading the exact same file again
    e.target.value = "";

    const form = new FormData();
    form.append("video", file);
    const res = await fetch(`${API}/api/upload`, { method: "POST", body: form });
    const data = await res.json();
    if (data.path) {
      setSelectedVideo(data.path);
      
      // Force load the new video frame (crucial if the path is the same as before)
      fetch(`${API}/api/load_video`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_path: data.path }),
      }).catch(() => {});

      const vids = await fetch(`${API}/api/videos`).then((r) => r.json());
      setVideos(vids);
    }
  };

  // ── Zone handlers ──
  const handleVideoClick = useCallback((e) => {
    if (!drawingZone) return;
    if (zonePoints.length >= 4) return; // Limit to 4 points

    const rect = videoContainerRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;

    setZonePoints((prev) => [...prev, { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) }]);
  }, [drawingZone, zonePoints]);

  const handleSaveZone = () => {
    if (zonePoints.length < 3) return;
    setDrawingZone(false);
    setZoneSaved(true);
    fetch(`${API}/api/zone`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ points: zonePoints }),
    });
  };

  const handleClearZone = () => {
    setZonePoints([]);
    setDrawingZone(false);
    setZoneSaved(false);
    fetch(`${API}/api/zone`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ points: [] }),
    });
  };

  const handleStartDrawing = () => {
    setZonePoints([]);
    setDrawingZone(true);
    setZoneSaved(false);
  };

  const handleExportLogs = () => {
    if (events.length === 0) return;
    const header = "Timestamp,VideoTime(s),Type,Message\n";
    const rows = events.map(e => `"${e.timestamp}","${e.video_time}","${e.type}","${e.message}"`).join("\n");
    const csvContent = header + rows;
    
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `loitering_logs_${new Date().toISOString().split("T")[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleReset = async () => {
    try {
      await fetch(`${API}/api/reset`, { method: "POST" });
    } catch (err) {
      console.error(err);
    }
    handleConfigChange("presence_threshold", 10);
    handleConfigChange("loitering_threshold", 30);
    handleClearZone();
    setSelectedVideo("");
    setShowPhoneInput(false);
    setPhoneUrl("");
    setEvents([]);
  };

  const isRunning = stats.status === "running" || stats.status === "loading" || stats.status === "starting";

  const getEventClass = (type) => {
    if (type === "loitering") return "event-loitering";
    if (type === "enter") return "event-enter";
    if (type === "error") return "event-error";
    return "event-info";
  };

  // Build SVG polygon points string
  const svgPolyPoints = zonePoints
    .map((p) => `${p.x * 100}%,${p.y * 100}%`)
    .join(" ");

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-left">
          <div className={`status-dot ${connected ? "connected" : ""}`} />
          <h1>Suspicious Behavior Detection</h1>
        </div>
        <div className="header-right">
          <span className={`status-badge ${stats.status}`}>
            {stats.status}
          </span>
        </div>
      </header>

      {/* ── Main Content ── */}
      <main className="main">
        {/* Video Feed */}
        <section className="video-section">
          <div
            className="video-container"
            ref={videoContainerRef}
            onClick={handleVideoClick}
            style={{ cursor: drawingZone ? "crosshair" : "default" }}
          >
            {selectedVideo ? (
              <img
                key={selectedVideo}
                src={`${API}/api/video_feed`}
                alt="Detection Feed"
                className="video-feed"
              />
            ) : (
              <div className="video-placeholder">
                <svg viewBox="0 0 24 24" width="48" height="48" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect>
                  <line x1="7" y1="2" x2="7" y2="22"></line>
                  <line x1="17" y1="2" x2="17" y2="22"></line>
                  <line x1="2" y1="12" x2="22" y2="12"></line>
                  <line x1="2" y1="7" x2="7" y2="7"></line>
                  <line x1="2" y1="17" x2="7" y2="17"></line>
                  <line x1="17" y1="17" x2="22" y2="17"></line>
                  <line x1="17" y1="7" x2="22" y2="7"></line>
                </svg>
                <p>Select or upload a video to begin monitoring</p>
              </div>
            )}
            {/* Zone overlay (drawn while editing, before save) */}
            {drawingZone && zonePoints.length > 0 && (
              <svg className="zone-overlay">
                {/* Lines between points */}
                {zonePoints.map((p, i) => {
                  const next = zonePoints[(i + 1) % zonePoints.length];
                  if (i === zonePoints.length - 1 && zonePoints.length < 3) return null;
                  return (
                    <line
                      key={i}
                      x1={`${p.x * 100}%`} y1={`${p.y * 100}%`}
                      x2={`${next.x * 100}%`} y2={`${next.y * 100}%`}
                      stroke="#00bcd4" strokeWidth="2" strokeDasharray="6 3"
                    />
                  );
                })}
                {/* Points */}
                {zonePoints.map((p, i) => (
                  <circle
                    key={i}
                    cx={`${p.x * 100}%`} cy={`${p.y * 100}%`}
                    r="5" fill="#00bcd4" stroke="#0d0d0d" strokeWidth="2"
                  />
                ))}
              </svg>
            )}

            {/* Drawing mode indicator */}
            {drawingZone && (
              <div className="zone-hint">
                Click to place points ({zonePoints.length}/4)
                {zonePoints.length === 4 && " — ready to save"}
              </div>
            )}
          </div>

          {/* Stats Bar */}
          <div className="stats-bar">
            <div className="stat">
              <span className="stat-label">FPS</span>
              <span className="stat-value">{stats.fps}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Frame</span>
              <span className="stat-value">{stats.frame_num}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Tracked</span>
              <span className="stat-value">{stats.total_tracked}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Alerts</span>
              <span className={`stat-value ${stats.active_alerts > 0 ? "alert-active" : ""}`}>
                {stats.active_alerts}
              </span>
            </div>
          </div>
        </section>

        {/* Sidebar */}
        <aside className="sidebar">
          {/* Controls */}
          <div className="panel">
            <div className="panel-header">
              <h2 className="panel-title">Controls</h2>
              <button 
                className="btn-icon" 
                onClick={handleReset} 
                title="Reset Settings & Logs"
              >
                <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path>
                  <path d="M3 3v5h5"></path>
                </svg>
              </button>
            </div>

            <div className="control-group">
              <label className="control-label">
                <span>Video Source</span>
              </label>

              <select
                className="control-select"
                value={selectedVideo !== '0' && selectedVideo !== phoneUrl ? selectedVideo : ""}
                onChange={(e) => { setSelectedVideo(e.target.value); setShowPhoneInput(false); }}
                disabled={isRunning}
              >
                <option value="">Select a local video...</option>
                {videos.map((v, i) => (
                  <option key={i} value={v.path}>{v.name}</option>
                ))}
              </select>

              <label className="upload-btn" data-disabled={isRunning}>
                <input
                  type="file"
                  accept="video/*"
                  onChange={handleUpload}
                  hidden
                  disabled={isRunning}
                />
                Upload Video
              </label>
            </div>

            {/* Zone Controls */}
            <div className="control-group">
              <label className="control-label">
                Monitoring Zone
                {zoneSaved && <span className="zone-status"> ✓ Active</span>}
              </label>
              <div className="zone-buttons">
                <div className="zone-buttons-row">
                  {!drawingZone ? (
                    <button className="btn btn-zone" onClick={handleStartDrawing}>
                      ✎ Draw Zone
                    </button>
                  ) : (
                    <button
                      className="btn btn-zone btn-save"
                      onClick={handleSaveZone}
                      disabled={zonePoints.length !== 4}
                    >
                      ✓ Save ({zonePoints.length}/4 pts)
                    </button>
                  )}
                  {(zonePoints.length > 0 || zoneSaved) && (
                    <button className="btn btn-zone btn-clear" onClick={handleClearZone}>
                      ✕ Clear
                    </button>
                  )}
                </div>
              </div>
            </div>

            <div className="control-group">
              <label className="control-label">
                Presence: <strong>{config.presence_threshold}s</strong>
              </label>
              <input
                type="range"
                min="1" max="60" step="1"
                value={config.presence_threshold}
                onChange={(e) =>
                  handleConfigChange("presence_threshold", +e.target.value)
                }
                className="control-slider"
              />
            </div>

            <div className="control-group">
              <label className="control-label">
                Loitering: <strong>{config.loitering_threshold}s</strong>
              </label>
              <input
                type="range"
                min="5" max="120" step="1"
                value={config.loitering_threshold}
                onChange={(e) =>
                  handleConfigChange("loitering_threshold", +e.target.value)
                }
                className="control-slider"
              />
            </div>

            <div className="control-actions">
              {!isRunning ? (
                <button
                  className="btn btn-start"
                  onClick={handleStart}
                  disabled={!selectedVideo}
                >
                  ▶ Start Detection
                </button>
              ) : (
                <button className="btn btn-stop" onClick={handleStop}>
                  ■ Stop
                </button>
              )}
            </div>
          </div>

          {/* Event Log */}
          <div className="panel panel-log">
            <div className="panel-header">
              <h2 className="panel-title">Event Log</h2>
              <button 
                className="btn-export" 
                onClick={handleExportLogs} 
                disabled={events.length === 0}
                title="Download logs as CSV"
              >
                ↓ Export CSV
              </button>
            </div>
            <div className="event-log" ref={logRef}>
              {events.length === 0 && (
                <div className="event-empty">No events yet</div>
              )}
              {events.map((evt, i) => (
                <div key={i} className={`event-item ${getEventClass(evt.type)}`}>
                  <span className="event-time">{evt.timestamp}</span>
                  <span className="event-msg">{evt.message}</span>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;
