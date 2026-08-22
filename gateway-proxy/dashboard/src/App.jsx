import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import styles from './App.module.scss'

const API_DASHBOARD = '/v1/dashboard'
const API_LOGS = '/v1/logs?limit=200'

// ─── helpers ──────────────────────────────────────────────────────────

function fmt(n) {
  if (!Number.isFinite(n)) return '0'
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + 'B'
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + 'M'
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(2) + 'K'
  return n.toLocaleString()
}

function fmtMs(ms) {
  if (!ms) return '<1ms'
  if (ms < 1) return '<1ms'
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

function fmtTs(ts) {
  if (!ts) return '-'
  return new Date(ts * 1000).toLocaleString()
}

function latencyTone(ms) {
  if (ms < 2000) return 'fast'
  if (ms < 8000) return 'mid'
  return 'slow'
}

// ─── spring count-up ──────────────────────────────────────────────────

const easeOutExpo = (x) => (x === 1 ? 1 : 1 - Math.pow(2, -10 * x))

function useCountUp(target, format) {
  const [display, setDisplay] = useState(() => format(0))
  const currentRef = useRef(0)

  useEffect(() => {
    if (!Number.isFinite(target)) {
      currentRef.current = target
      setDisplay(format(target))
      return
    }
    const from = currentRef.current
    if (from === target) {
      setDisplay(format(target))
      return
    }
    let raf = 0
    const duration = 900
    const start = performance.now()
    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration)
      const value = from + (target - from) * easeOutExpo(t)
      currentRef.current = value
      setDisplay(format(value))
      if (t < 1) {
        raf = requestAnimationFrame(tick)
      } else {
        currentRef.current = target
        setDisplay(format(target))
      }
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target])

  return display
}

const formatInt = (n) => Math.round(n).toLocaleString()

function AnimatedNumber({ value, format }) {
  const display = useCountUp(value, format)
  return <>{display}</>
}

// ─── mini sparkline (in stat card) ─────────────────────────────────────

function MiniSparkline({ data, color }) {
  if (!data || data.length === 0) return null
  const w = 80
  const h = 24
  const max = Math.max(1, ...data)
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1 || 1)) * w
    const y = h - (v / max) * (h - 4) - 2
    return `${x},${y}`
  })
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className={styles.sparkline} preserveAspectRatio="none">
      <polyline points={pts.join(' ')} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  )
}

// ─── stat card ────────────────────────────────────────────────────────

function StatCard({ label, value, hint, tone = 'default', sparkData, sparkColor }) {
  return (
    <div className={`${styles.statCard} ${styles[`tone-${tone}`] || ''}`}>
      <div className={styles.statGlow} />
      <div className={styles.statContent}>
        <div className={styles.statTop}>
          <span className={styles.statLabel}>{label}</span>
          {sparkData && <MiniSparkline data={sparkData} color={sparkColor} />}
        </div>
        <div className={styles.statValue}>{value}</div>
        {hint ? <div className={styles.statHint}>{hint}</div> : null}
      </div>
    </div>
  )
}

// ─── bar chart ────────────────────────────────────────────────────────

function BarChart({ data }) {
  const BAR_COLORS = ['#6366f1', '#10b981', '#a855f7', '#f59e0b', '#ef4444', '#06b6d4']
  const max = Math.max(1, ...data.map((d) => d.value))
  const total = Math.max(1, data.reduce((s, d) => s + d.value, 0))
  return (
    <div className={styles.barChart}>
      {data.map((d, i) => {
        const color = BAR_COLORS[i % BAR_COLORS.length]
        const pct = (d.value / total) * 100
        return (
          <div key={i} className={styles.barRow}>
            <div className={styles.barLabel} title={d.label}>
              <span className={styles.barDot} style={{ backgroundColor: color }} />
              {d.label}
            </div>
            <div className={styles.barTrack}>
              <div
                className={styles.barFill}
                style={{
                  width: `${Math.min(100, (d.value / max) * 100)}%`,
                  background: `linear-gradient(90deg, ${color}, ${color}99)`,
                }}
              />
            </div>
            <div className={styles.barValue}>{fmt(d.value)}</div>
            <div className={styles.barPct}>{pct.toFixed(1)}%</div>
          </div>
        )
      })}
    </div>
  )
}

// ─── line chart ───────────────────────────────────────────────────────

function LineChart({ points }) {
  const [hoverIdx, setHoverIdx] = useState(null)
  if (!points || !points.length) return null

  const W = 900, H = 280, PAD_X = 56, PAD_TOP = 28, PAD_BOTTOM = 36
  const maxTokens = Math.max(1, ...points.map((p) => p.completion + p.prompt))
  const maxReq = Math.max(1, ...points.map((p) => p.requests))

  const x = (i) => PAD_X + (i / Math.max(1, points.length - 1)) * (W - PAD_X * 2)
  const yTok = (v) => H - PAD_BOTTOM - (v / maxTokens) * (H - PAD_TOP - PAD_BOTTOM)
  const yReq = (v) => H - PAD_BOTTOM - (v / maxReq) * (H - PAD_TOP - PAD_BOTTOM)

  const smoothLine = (vals, toY) => {
    if (vals.length < 3) {
      return vals.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${toY(v).toFixed(1)}`).join(' ')
    }
    const pts = vals.map((v, i) => [x(i), toY(v)])
    let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[Math.max(0, i - 1)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[Math.min(pts.length - 1, i + 2)]
      const c1x = p1[0] + (p2[0] - p0[0]) / 6, c1y = p1[1] + (p2[1] - p0[1]) / 6
      const c2x = p2[0] - (p3[0] - p1[0]) / 6, c2y = p2[1] - (p3[1] - p1[1]) / 6
      d += ` C${c1x.toFixed(1)},${c1y.toFixed(1)} ${c2x.toFixed(1)},${c2y.toFixed(1)} ${p2[0].toFixed(1)},${p2[1].toFixed(1)}`
    }
    return d
  }

  const area = (vals, toY) =>
    `${smoothLine(vals, toY)} L${x(vals.length - 1).toFixed(1)},${H - PAD_BOTTOM} L${x(0).toFixed(1)},${H - PAD_BOTTOM} Z`

  const completionVals = points.map((p) => p.completion)
  const promptVals = points.map((p) => p.prompt)
  const cachedVals = points.map((p) => p.cached)
  const reqVals = points.map((p) => p.requests)
  const hovered = hoverIdx !== null ? points[hoverIdx] : null

  return (
    <div className={styles.lineChartWrap}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className={styles.lineChart}
        onMouseLeave={() => setHoverIdx(null)}
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect()
          const relX = ((e.clientX - rect.left) / rect.width) * W
          const frac = (relX - PAD_X) / Math.max(1, W - PAD_X * 2)
          setHoverIdx(Math.max(0, Math.min(points.length - 1, Math.round(frac * (points.length - 1)))))
        }}
      >
        <defs>
          <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" className={styles.areaStopTop} />
            <stop offset="100%" className={styles.areaStopBottom} />
          </linearGradient>
          <filter id="lineGlow">
            <feGaussianBlur stdDeviation="2" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const y = PAD_TOP + f * (H - PAD_TOP - PAD_BOTTOM)
          return (
            <g key={f}>
              <line x1={PAD_X} x2={W - PAD_X} y1={y} y2={y} className={styles.gridLine} />
              <text x={PAD_X - 10} y={y + 4} textAnchor="end" className={styles.gridLabel}>
                {fmt(maxTokens * (1 - f))}
              </text>
            </g>
          )
        })}
        <path d={area(completionVals, yTok)} fill="url(#areaGrad)" stroke="none" />
        <path d={smoothLine(completionVals, yTok)} className={styles.lineOutput} fill="none" filter="url(#lineGlow)" />
        <path d={smoothLine(promptVals, yTok)} className={styles.lineInput} fill="none" />
        <path d={smoothLine(cachedVals, yTok)} className={styles.lineCached} fill="none" />
        <path d={smoothLine(reqVals, yReq)} className={styles.lineRequests} fill="none" strokeDasharray="4 3" />
        {points.map((p, i) =>
          i % Math.ceil(points.length / 8) === 0 ? (
            <text key={i} x={x(i)} y={H - 12} className={styles.gridLabel} textAnchor="middle">{p.label || ''}</text>
          ) : null
        )}
        {hovered && hoverIdx !== null && (
          <g className={styles.crosshair}>
            <line x1={x(hoverIdx)} x2={x(hoverIdx)} y1={PAD_TOP} y2={H - PAD_BOTTOM} className={styles.crosshairLine} />
            <circle cx={x(hoverIdx)} cy={yTok(hovered.completion)} r={5} className={styles.dotOutput} />
            <circle cx={x(hoverIdx)} cy={yTok(hovered.prompt)} r={3.5} className={styles.dotInput} />
            <circle cx={x(hoverIdx)} cy={yTok(hovered.cached)} r={3.5} className={styles.dotCached} />
          </g>
        )}
      </svg>
      <div className={styles.chartFooter}>
        <div className={styles.legend}>
          <span className={styles.legendItem}><span className={`${styles.legendDot} ${styles.legendOutput}`} /> Completion</span>
          <span className={styles.legendItem}><span className={`${styles.legendDot} ${styles.legendInput}`} /> Prompt</span>
          <span className={styles.legendItem}><span className={`${styles.legendDot} ${styles.legendCached}`} /> Cached</span>
          <span className={styles.legendItem}><span className={`${styles.legendDot} ${styles.legendRequests}`} /> Requests</span>
        </div>
        <div className={`${styles.tooltip} ${hovered ? styles.tooltipVisible : ''}`}>
          {hovered ? (
            <>
              <span className={styles.tooltipTime}>{hovered.label || ''}</span>
              <span className={styles.tooltipRow}><span className={`${styles.legendDot} ${styles.legendOutput}`} />{fmt(hovered.completion)}</span>
              <span className={styles.tooltipRow}><span className={`${styles.legendDot} ${styles.legendInput}`} />{fmt(hovered.prompt)}</span>
              <span className={styles.tooltipRow}><span className={`${styles.legendDot} ${styles.legendCached}`} />{fmt(hovered.cached)}</span>
              <span className={styles.tooltipRow}><span className={`${styles.legendDot} ${styles.legendRequests}`} />{fmt(hovered.requests)}</span>
            </>
          ) : <span className={styles.tooltipHint}>&nbsp;</span>}
        </div>
      </div>
    </div>
  )
}

// ─── donut ────────────────────────────────────────────────────────────

function DonutChart({ segments, centerLabel, centerValue }) {
  const total = Math.max(1, segments.reduce((s, x) => s + x.value, 0))
  const R = 54
  const CIRC = 2 * Math.PI * R
  const slices = segments.map((s, i) => ({
    frac: s.value / total,
    offset: (-segments.slice(0, i).reduce((a, x) => a + x.value, 0) / total) * CIRC,
    color: s.color,
    label: s.label,
  }))
  return (
    <div className={styles.donutWrap}>
      <div className={styles.donutSvgWrap}>
        <svg viewBox="0 0 140 140" className={styles.donutSvg}>
          <circle cx="70" cy="70" r={R} className={styles.donutTrack} />
          {slices.map((sl, i) => sl.frac <= 0 ? null : (
            <circle key={i} cx="70" cy="70" r={R} className={styles.donutSegment} stroke={sl.color}
              strokeDasharray={`${Math.max(0, sl.frac * CIRC - 2)} ${CIRC - sl.frac * CIRC + 2}`} strokeDashoffset={sl.offset} />
          ))}
        </svg>
        <div className={styles.donutCenter}>
          <div className={styles.donutValue}>{centerValue}</div>
          <div className={styles.donutLabel}>{centerLabel}</div>
        </div>
      </div>
      <div className={styles.donutLegend}>
        {segments.map((s, i) => {
          const pct = (s.value / total) * 100
          return (
            <div key={i} className={styles.donutLegendRow}>
              <span className={styles.barDot} style={{ backgroundColor: s.color }} />
              <span className={styles.donutLegendLabel}>{s.label}</span>
              <span className={styles.donutLegendPct}>{pct.toFixed(1)}%</span>
              <span className={styles.donutLegendVal}>{fmt(s.value)}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ─── heatmap ──────────────────────────────────────────────────────────

function Heatmap({ series }) {
  if (!series || !series.length) return null
  const maxReq = Math.max(1, ...series.map((p) => p.requests))
  return (
    <div className={styles.heatmapWrap}>
      <div className={styles.heatmapGrid} style={{ gridTemplateColumns: `repeat(${series.length}, 1fr)` }}>
        {series.map((p, i) => {
          const intensity = Math.min(1, (p.requests / maxReq) ** 0.45)
          return (
            <div key={i} className={styles.heatCell} style={{ opacity: 0.12 + 0.88 * intensity }}
              title={`${p.label || ''} · ${p.requests} req · ${fmt(p.prompt + p.completion)} tok`} />
          )
        })}
      </div>
      <div className={styles.heatmapLegend}>
        <span className={styles.heatLabel}>Quiet</span>
        {[0.2, 0.45, 0.7, 0.9].map((v) => (
          <div key={v} className={styles.heatLegendCell} style={{ opacity: 0.12 + 0.88 * v }} />
        ))}
        <span className={styles.heatLabel}>Busy</span>
      </div>
    </div>
  )
}

// ─── scatter ──────────────────────────────────────────────────────────

function LatencyScatter({ records }) {
  if (!records || !records.length) return null
  const W = 560, H = 260, PAD_X = 46, PAD_TOP = 18, PAD_BOTTOM = 34
  const lat = records.map((r) => r.duration_ms || 0).filter((v) => v >= 0)
  if (!lat.length) return null
  const maxLat = Math.max(1, ...lat)
  const maxTok = Math.max(1, ...records.map((r) => (r.prompt_tokens + r.completion_tokens) || 1))
  const tMin = Math.min(...records.map((r) => r.ts))
  const tMax = Math.max(tMin + 1, ...records.map((r) => r.ts))
  const x = (ts) => PAD_X + ((ts - tMin) / Math.max(1, tMax - tMin)) * (W - PAD_X - 18)
  const y = (ms) => H - PAD_BOTTOM - (ms / maxLat) * (H - PAD_TOP - PAD_BOTTOM)
  const r = (tok) => 3 + 5 * Math.sqrt((tok || 1) / maxTok)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={styles.scatterSvg}>
      {[0, 0.5, 1].map((f) => {
        const yy = PAD_TOP + f * (H - PAD_TOP - PAD_BOTTOM)
        return (
          <g key={f}>
            <line x1={PAD_X} x2={W - 14} y1={yy} y2={yy} className={styles.gridLine} />
            <text x={PAD_X - 7} y={yy + 3.5} textAnchor="end" className={styles.gridLabel}>{fmtMs(maxLat * (1 - f))}</text>
          </g>
        )
      })}
      {records.map((rec, i) => (
        <circle key={i} cx={x(rec.ts)} cy={y(rec.duration_ms || 0)} r={r(rec.prompt_tokens + rec.completion_tokens)}
          className={rec.error ? styles.scatterDotFail : styles.scatterDot}>
          <title>{`${rec.model} · ${fmtMs(rec.duration_ms)} · ${fmt(rec.prompt_tokens + rec.completion_tokens)} tok`}</title>
        </circle>
      ))}
    </svg>
  )
}

// ─── success ring ─────────────────────────────────────────────────────

function SuccessRing({ rate }) {
  const clamped = Math.max(0, Math.min(100, rate))
  const R = 15.5, C = 2 * Math.PI * R
  const color = clamped >= 99 ? '#10b981' : clamped >= 95 ? '#6366f1' : clamped >= 90 ? '#f59e0b' : '#ef4444'
  return (
    <span className={styles.ring} title={`${clamped.toFixed(1)}%`}>
      <svg viewBox="0 0 40 40" className={styles.ringSvg}>
        <circle cx="20" cy="20" r={R} className={styles.ringTrack} />
        <circle cx="20" cy="20" r={R} className={styles.ringFill} stroke={color}
          strokeDasharray={`${(clamped / 100) * C} ${C}`} />
      </svg>
      <span className={styles.ringText} style={{ color }}>{Math.round(clamped)}</span>
    </span>
  )
}

// ─── token numbers ────────────────────────────────────────────────────

function TokenNumbers({ input, output, reasoning, cached }) {
  const items = [
    { key: 'input', label: 'In', value: input, cls: styles.tnInput },
    { key: 'output', label: 'Out', value: output, cls: styles.tnOutput },
    { key: 'reasoning', label: 'Think', value: reasoning, cls: styles.tnReasoning },
    { key: 'cached', label: 'Cache', value: cached, cls: styles.tnCached },
  ]
  const sum = items.reduce((s, it) => s + Math.max(0, it.value), 0)
  return (
    <span className={styles.tokenNums}>
      {sum > 0 && (
        <span className={styles.tokenStack} aria-hidden="true">
          {items.map((it) => it.value > 0 ? (
            <span key={it.key} className={`${styles.tokenStackSeg} ${it.cls}`} style={{ width: `${(Math.max(0, it.value) / sum) * 100}%` }} />
          ) : null)}
        </span>
      )}
      <span className={styles.tokenChips}>
        {items.map((it) => (
          <span key={it.key} className={`${styles.tokenChip} ${it.cls} ${it.value <= 0 ? styles.tokenChipZero : ''}`}
            title={`${it.label}: ${Math.max(0, it.value).toLocaleString()}`}>
            <span className={styles.tokenChipDot} aria-hidden="true" />
            {fmt(it.value)}
          </span>
        ))}
      </span>
    </span>
  )
}

// ─── main app ─────────────────────────────────────────────────────────

export default function App() {
  const [data, setData] = useState(null)
  const [logs, setLogs] = useState([])
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [tab, setTab] = useState('usage')
  const [theme, setTheme] = useState('dark')
  const timerRef = useRef(null)

  useEffect(() => {
    const saved = localStorage.getItem('dashboard-theme') || 'dark'
    setTheme(saved)
    document.documentElement.setAttribute('data-theme', saved)
  }, [])

  const toggleTheme = () => {
    const next = theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    document.documentElement.setAttribute('data-theme', next)
    localStorage.setItem('dashboard-theme', next)
  }

  const fetchAll = useCallback(async () => {
    try {
      const [dashResp, logsResp] = await Promise.all([fetch(API_DASHBOARD), fetch(API_LOGS)])
      if (!dashResp.ok) throw new Error(`Dashboard HTTP ${dashResp.status}`)
      const dash = await dashResp.json()
      setData(dash)
      if (logsResp.ok) {
        const logData = await logsResp.json()
        setLogs(logData.lines || [])
      }
      setError(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchAll() }, [fetchAll])

  useEffect(() => {
    if (!autoRefresh) {
      if (timerRef.current) clearInterval(timerRef.current)
      return
    }
    timerRef.current = setInterval(fetchAll, 5000)
    return () => clearInterval(timerRef.current)
  }, [autoRefresh, fetchAll])

  const t = useMemo(() => data?.totals || {}, [data])
  const ts = useMemo(() => data?.time_series || { labels: [], prompt: [], completion: [], cached: [], requests: [] }, [data])

  if (loading) {
    return (
      <div className={styles.loadingWrap}>
        <div className={styles.loaderOrb} />
        <div className={styles.loadingText}>Initializing</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className={styles.loadingWrap}>
        <div className={styles.errorBox}><span className={styles.errorIcon}>!</span> {error}</div>
        <button className={styles.btn} onClick={fetchAll}>Retry</button>
      </div>
    )
  }

  const uptimeMin = Math.round((t.uptime_s || 0) / 60)
  const successRate = t.requests > 0 ? ((t.requests - t.errors) / t.requests) * 100 : -1

  const donutSegments = [
    { label: 'Prompt', value: t.prompt_tokens, color: '#10b981' },
    { label: 'Completion', value: t.completion_tokens, color: '#a855f7' },
    { label: 'Reasoning', value: t.reasoning_tokens, color: '#f59e0b' },
    { label: 'Cached', value: t.cached_tokens, color: '#06b6d4' },
  ].filter((s) => s.value > 0)

  const modelBars = (data?.per_model || []).slice(0, 6).map((m) => ({
    label: m.model,
    value: m.prompt_tokens + m.completion_tokens,
  }))

  const chartPoints = ts.labels.map((label, i) => ({
    label,
    prompt: ts.prompt[i] || 0,
    completion: ts.completion[i] || 0,
    cached: ts.cached[i] || 0,
    requests: ts.requests[i] || 0,
  })).filter((p) => p.requests > 0 || p.prompt > 0 || p.completion > 0)

  // sparkline data (non-zero slices of time series)
  const sparkReq = ts.requests.filter((v) => v > 0)
  const sparkCompletion = ts.completion.filter((v) => v > 0)
  const sparkPrompt = ts.prompt.filter((v) => v > 0)
  const sparkCached = ts.cached.filter((v) => v > 0)

  return (
    <div className={styles.container}>
      {/* Ambient background */}
      <div className={styles.ambient} aria-hidden="true">
        <div className={styles.ambientOrb1} />
        <div className={styles.ambientOrb2} />
        <div className={styles.ambientOrb3} />
      </div>

      {/* Header */}
      <header className={styles.hero}>
        <div className={styles.heroLeft}>
          <div className={styles.liveDot} />
          <div>
            <h1 className={styles.title}>Fionn</h1>
            <p className={styles.subtitle}>
              {uptimeMin}m uptime · {t.requests || 0} requests served
            </p>
          </div>
        </div>
        <div className={styles.controls}>
          <div className={styles.tabSwitch}>
            <button className={`${styles.tabBtn} ${tab === 'usage' ? styles.tabActive : ''}`} onClick={() => setTab('usage')}>
              <span className={styles.tabDot} /> Usage
            </button>
            <button className={`${styles.tabBtn} ${tab === 'logs' ? styles.tabActive : ''}`} onClick={() => setTab('logs')}>
              <span className={styles.tabDotLogs} /> Logs
            </button>
            <div className={styles.tabIndicator} style={{ transform: tab === 'logs' ? 'translateX(100%)' : 'translateX(0)' }} />
          </div>
          <button className={styles.iconBtn} onClick={toggleTheme} title="Toggle theme">
            {theme === 'dark' ? (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="5" /><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
              </svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
              </svg>
            )}
          </button>
          <button className={styles.iconBtn} onClick={fetchAll} title="Refresh">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M23 4v6h-6M1 20v-6h6" /><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
            </svg>
          </button>
          <label className={styles.autoToggle}>
            <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} />
            <span className={styles.autoTrack}><span className={styles.autoThumb} /></span>
            <span className={styles.autoLabel}>5s</span>
          </label>
        </div>
      </header>

      {tab === 'usage' && (
        <>
          {/* Bento stat grid */}
          <div className={styles.bento}>
            <StatCard
              label="Total Requests"
              value={<AnimatedNumber value={t.requests} format={fmt} />}
              hint={<><span className={styles.hintDot} style={{ background: 'var(--success-color)' }} /> {t.errors} errors</>}
              tone="indigo"
              sparkData={sparkReq}
              sparkColor="#6366f1"
            />
            <StatCard
              label="Throughput"
              value={<AnimatedNumber value={t.tps} format={formatInt} />}
              hint="tokens / sec"
              tone="emerald"
              sparkData={sparkCompletion}
              sparkColor="#10b981"
            />
            <StatCard
              label="Tokens In"
              value={<AnimatedNumber value={t.prompt_tokens} format={fmt} />}
              hint="prompt"
              tone="sky"
              sparkData={sparkPrompt}
              sparkColor="#0ea5e9"
            />
            <StatCard
              label="Tokens Out"
              value={<AnimatedNumber value={t.completion_tokens} format={fmt} />}
              hint="completion"
              tone="violet"
              sparkData={sparkCompletion}
              sparkColor="#a855f7"
            />
            <StatCard
              label="Cached"
              value={<AnimatedNumber value={t.cached_tokens} format={fmt} />}
              hint="cache read"
              tone="cyan"
              sparkData={sparkCached}
              sparkColor="#06b6d4"
            />
            <StatCard
              label="Errors"
              value={<AnimatedNumber value={t.errors} format={formatInt} />}
              hint={successRate >= 0 ? (
                <span className={styles.successHint}>
                  <SuccessRing rate={successRate} />
                  {successRate.toFixed(1)}% success
                </span>
              ) : undefined}
              tone={t.errors > 0 ? 'rose' : 'slate'}
            />
          </div>

          {/* Main chart */}
          <div className={styles.glassCard}>
            <div className={styles.cardHeader}>
              <span className={styles.sectionTitle}>Token Throughput</span>
              <span className={styles.windowNote}>Last 60 min</span>
            </div>
            {chartPoints.length > 0 ? <LineChart points={chartPoints} /> : <div className={styles.emptyState}>No data yet</div>}
          </div>

          {/* Viz bento */}
          <div className={styles.vizBento}>
            <div className={`${styles.glassCard} ${styles.span1}`}>
              <div className={styles.cardHeader}>
                <span className={styles.sectionTitle}>Composition</span>
              </div>
              {donutSegments.length > 0 ? (
                <DonutChart segments={donutSegments} centerLabel="Tokens" centerValue={fmt(t.prompt_tokens + t.completion_tokens)} />
              ) : <div className={styles.emptyState}>No data</div>}
            </div>
            <div className={`${styles.glassCard} ${styles.span1}`}>
              <div className={styles.cardHeader}>
                <span className={styles.sectionTitle}>Activity</span>
              </div>
              {chartPoints.length > 0 ? <Heatmap series={chartPoints} /> : <div className={styles.emptyState}>No data</div>}
            </div>
            <div className={`${styles.glassCard} ${styles.span2}`}>
              <div className={styles.cardHeader}>
                <span className={styles.sectionTitle}>Latency / Tokens</span>
              </div>
              {(data?.recent || []).length > 0 ? <LatencyScatter records={data.recent} /> : <div className={styles.emptyState}>No records</div>}
            </div>
          </div>

          {/* Per model */}
          <div className={styles.glassCard}>
            <div className={styles.cardHeader}>
              <span className={styles.sectionTitle}>Per Model</span>
            </div>
            {modelBars.length > 0 ? (
              <>
                <BarChart data={modelBars} />
                <div className={styles.tableWrap}>
                  <table className={styles.table}>
                    <thead>
                      <tr><th>Model</th><th>Reqs</th><th>Failed</th><th>In</th><th>Out</th><th>Cached</th><th>Total</th></tr>
                    </thead>
                    <tbody>
                      {(data?.per_model || []).map((m, i) => (
                        <tr key={i}>
                          <td className={styles.keyCell} title={m.model}>{m.model}</td>
                          <td>{fmt(m.requests)}</td>
                          <td className={m.errors > 0 ? styles.failedCell : ''}>{fmt(m.errors)}</td>
                          <td>{fmt(m.prompt_tokens)}</td>
                          <td>{fmt(m.completion_tokens)}</td>
                          <td>{fmt(m.cached_tokens)}</td>
                          <td className={styles.totalCell}>{fmt(m.prompt_tokens + m.completion_tokens)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : <div className={styles.emptyState}>No data</div>}
          </div>

          {/* Per caller */}
          <div className={styles.glassCard}>
            <div className={styles.cardHeader}><span className={styles.sectionTitle}>Per Caller</span></div>
            {(data?.per_caller || []).length > 0 ? (
              <div className={styles.tableWrap}>
                <table className={styles.table}>
                  <thead>
                    <tr><th>Caller</th><th>Reqs</th><th>Failed</th><th>In</th><th>Out</th><th>Cached</th></tr>
                  </thead>
                  <tbody>
                    {(data?.per_caller || []).map((c, i) => (
                      <tr key={i}>
                        <td className={styles.keyCell}>{c.caller}</td>
                        <td>{fmt(c.requests)}</td>
                        <td className={c.errors > 0 ? styles.failedCell : ''}>{fmt(c.errors)}</td>
                        <td>{fmt(c.prompt_tokens)}</td>
                        <td>{fmt(c.completion_tokens)}</td>
                        <td>{fmt(c.cached_tokens)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div className={styles.emptyState}>No data</div>}
          </div>

          {/* Recent requests */}
          <div className={styles.glassCard}>
            <div className={styles.cardHeader}>
              <span className={styles.sectionTitle}>
                Recent Requests
                {(data?.recent || []).length > 0 && <span className={styles.countBadge}>{(data?.recent || []).length}</span>}
              </span>
            </div>
            {(data?.recent || []).length > 0 ? (
              <div className={styles.tableWrap}>
                <table className={styles.table}>
                  <thead>
                    <tr><th>Time</th><th>Model</th><th>Endpoint</th><th>Tokens</th><th>TPS</th><th>Latency</th><th>Status</th></tr>
                  </thead>
                  <tbody>
                    {data.recent.map((r, i) => (
                      <tr key={i} className={r.error ? styles.failedRow : ''}>
                        <td className={styles.tsCell}>{fmtTs(r.ts)}</td>
                        <td><span className={styles.modelCell}>{r.model || '-'}</span></td>
                        <td className={styles.keyCell}>{r.endpoint || '-'}</td>
                        <td>
                          <span className={styles.tokensCell}>
                            <span className={styles.tokensValue}>{fmt(r.prompt_tokens + r.completion_tokens)}</span>
                            <TokenNumbers input={r.prompt_tokens} output={r.completion_tokens} reasoning={r.reasoning_tokens} cached={r.cached_tokens} />
                          </span>
                        </td>
                        <td><span className={styles.latencyPill}>{r.tps || 0}</span></td>
                        <td><span className={`${styles.latencyPill} ${styles[`latency-${latencyTone(r.duration_ms)}`]}`}>{fmtMs(r.duration_ms)}</span></td>
                        <td>{r.error ? <span className={styles.statusFail}>ERR</span> : <span className={styles.statusOk}>200</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div className={styles.emptyState}>No requests yet</div>}
          </div>
        </>
      )}

      {tab === 'logs' && (
        <div className={styles.glassCard}>
          <div className={styles.cardHeader}>
            <span className={styles.sectionTitle}>
              Proxy Logs
              {logs.length > 0 && <span className={styles.countBadge}>{logs.length}</span>}
            </span>
          </div>
          <div className={styles.logsContainer}>
            {logs.length > 0 ? logs.map((line, i) => {
              const isError = line.includes('[ERROR]') || line.includes('[WARNING]')
              const isWarn = line.includes('[WARNING]')
              return (
                <div key={i} className={`${styles.logLine} ${isError ? styles.logLineError : ''} ${isWarn ? styles.logLineWarn : ''}`}>
                  {line}
                </div>
              )
            }) : <div className={styles.emptyState}>No logs available</div>}
          </div>
        </div>
      )}
    </div>
  )
}
