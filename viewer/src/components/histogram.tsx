import React from 'react';
import Plot from 'react-plotly.js';
import { FeatureInfo } from '../types'

// "How likely is NEI, given how strongly the feature fires?" — claims are grouped into bands of activation
// strength at the verdict position (by share of claims: weakest 50%, next 30%, 15%, 4%, strongest 1%), plus the
// claims where it does not fire at all; each bar is the share of those claims whose gold label is NEI.
// Built from hist_by_label (raw counts per activation bin) and the fc summary, so no new data is needed.
const CUTS = [0.5, 0.8, 0.95, 0.99]
const NAMES = ['Weak', 'Medium', 'Strong', 'Very strong', 'Strongest']

type Band = {name: string, lo: number, hi: number, n: number, nei: number}

export function neiBands(data: FeatureInfo): Band[] {
  const hb = data.hist_by_label
  if (!hb || !data.fc) return []
  const nei = hb['NEI'] || [], dec = hb['SUP/REF'] || []
  const n = nei.map((v, i) => v + (dec[i] || 0))
  const T = n.reduce((a, b) => a + b, 0)
  const nTotal = Math.round(T / data.fc.frac_active)      // all verdict positions (active / fraction active)
  const bands: Band[] = []
  const off = nTotal - T, neiOff = data.fc.nei_base * nTotal - nei.reduce((a, b) => a + b, 0)
  if (off >= 50) bands.push({name: 'Does not fire', lo: 0, hi: 0, n: off, nei: neiOff})
  let cum = 0, start = 0, ci = 0
  const close = (s: number, e: number) => {
    const tot = n.slice(s, e + 1).reduce((a, b) => a + b, 0)
    if (tot > 0) bands.push({name: NAMES[Math.min(bands.filter((b) => b.hi > 0).length, NAMES.length - 1)],
                             lo: hb.edges[s], hi: hb.edges[e + 1], n: tot, nei: nei.slice(s, e + 1).reduce((a, b) => a + b, 0)})
  }
  for (let i = 0; i < n.length; i++) {
    cum += n[i]
    if (ci < CUTS.length && cum / T >= CUTS[ci]) {
      close(start, i); start = i + 1
      while (ci < CUTS.length && cum / T >= CUTS[ci]) ci++
    }
  }
  if (start < n.length) close(start, n.length - 1)
  return bands
}

const LABEL: {[n: string]: string} = {'Does not fire': 'does not fire', Weak: 'weak', Medium: 'medium', Strong: 'strong',
                                      'Very strong': 'very strong', Strongest: 'strongest'}

// one plain sentence that matches the bars (works for features linked to NEI and for "mirror" features)
export function bandSummary(data: FeatureInfo): string {
  const rs = nei_ranges(data)
  if (rs.length < 2 || !data.fc) return ''
  const r = (x: Range) => Math.round(100 * x.nei / x.n), lo = rs[0], hi = rs[rs.length - 1]
  const more = r(hi) > r(lo), fmt = (x: number) => Math.round(x).toLocaleString('en-US')
  const h = highlightRegion(data)
  const ext = more ? Math.max(...rs.map(r)) : Math.min(...rs.map(r))
  return `The stronger this feature fires, the ${more ? 'MORE' : 'LESS'} often the evidence is not enough (NEI): ` +
         `from ${r(lo)}% of claims in the weakest range ${more ? 'up' : 'down'} to ${ext}% when it fires strongly.` +
         (h ? (h.mirror ? ` From activation ${h.t.toFixed(2)} up, only ${Math.round(100 * h.nei / h.n)}% are NEI (${fmt(h.nei)} of ${fmt(h.n)} claims).`
                        : ` From activation ${h.t.toFixed(2)} up, most claims are NEI: ${Math.round(100 * h.nei / h.n)}% (${fmt(h.nei)} of ${fmt(h.n)}).`) : '')
}

// The result worth highlighting: the activation t from which, bin by bin up to the strongest, NEI is the majority
// of the claims (raw counts, so the share is the real one). For "mirror" features (AUC < 0.5) it is instead the
// activation from which NEI stays at most half of its dataset rate. Shown only if the region has >= 50 claims.
export function highlightRegion(data: FeatureInfo): {t: number, nei: number, n: number, mirror: boolean} | null {
  const hb = data.hist_by_label
  if (!hb || !data.fc) return null
  const nei = hb['NEI'] || [], dec = hb['SUP/REF'] || []
  const auc = data.fc.auc_nei_vs_dec_approx, mirror = auc !== null && auc !== undefined && auc < 0.5
  const ok = (k: number) => {
    const t = nei[k] + (dec[k] || 0)
    return t === 0 || (mirror ? nei[k] / t <= data.fc!.nei_base / 2 : nei[k] > (dec[k] || 0))
  }
  let i = nei.length
  while (i > 0 && ok(i - 1)) i--
  const sum = (a: number[]) => a.slice(i).reduce((x, y) => x + y, 0)
  const nN = sum(nei), n = nN + sum(dec)
  return n >= 50 ? {t: hb.edges[i], nei: nN, n, mirror} : null
}

// Equal-width activation ranges (the 40 histogram bins grouped into ~10); a range with fewer than MIN_N claims is
// merged into the next one, so the share of NEI shown on top of each bar is never based on a handful of claims.
const N_RANGES = 10, MIN_N = 30
type Range = {lo: number, hi: number, nei: number, n: number}
export function nei_ranges(data: FeatureInfo): Range[] {
  const hb = data.hist_by_label
  if (!hb) return []
  const nei = hb['NEI'] || [], dec = hb['SUP/REF'] || []
  const g = Math.max(1, Math.round(nei.length / N_RANGES))
  const out: Range[] = []
  let cur: Range | null = null
  for (let i = 0; i < nei.length; i++) {
    if (!cur) cur = {lo: hb.edges[i], hi: hb.edges[i + 1], nei: 0, n: 0}
    cur.nei += nei[i]; cur.n += nei[i] + (dec[i] || 0); cur.hi = hb.edges[i + 1]
    if ((i + 1) % g === 0 && cur.n >= MIN_N) { out.push(cur); cur = null }
  }
  if (cur && cur.n > 0) {
    if (cur.n < MIN_N && out.length) { const l = out[out.length - 1]; l.nei += cur.nei; l.n += cur.n; l.hi = cur.hi }
    else out.push(cur)
  }
  return out
}

// One bar per activation range: the share of its claims whose gold label is NEI. Bars where NEI is the majority are
// strong orange (for "mirror" features, AUC < 0.5: bars at most half the dataset rate are strong blue); the dataset
// rate is a discrete reference line. Claims where the feature does not fire are counted in the x-axis title.
const HistogramDisplay = ({ data }: {data: FeatureInfo}) => {
  const rs = nei_ranges(data)
  if (!rs.length || !data.fc) return <div className="text-sm text-gray-500">no distribution data for this feature</div>
  const base = data.fc.nei_base
  const auc = data.fc.auc_nei_vs_dec_approx, mirror = auc !== null && auc !== undefined && auc < 0.5
  const fmt = (x: number) => Math.round(x).toLocaleString('en-US')
  const short = (x: number) => x >= 10000 ? `${Math.round(x / 1000)}k` : x >= 1000 ? `${(x / 1000).toFixed(1)}k` : `${x}`
  const share = rs.map((r) => r.nei / r.n)
  const strong = share.map((v) => mirror ? v <= base / 2 : v > 0.5)
  const T = rs.reduce((a, r) => a + r.n, 0)
  const off = Math.round(T / data.fc.frac_active) - T
  const ticks = rs.map((r) => `${r.lo.toFixed(2)}–${r.hi.toFixed(2)}<br><span style="color:#888">n=${short(r.n)}</span>`)
  const traces = [{
    x: ticks, y: share.map((v) => 100 * v), type: 'bar',
    marker: {color: strong.map((s) => s ? (mirror ? '#2a78d6' : '#e8772e') : (mirror ? '#b9d1ee' : '#f6cfae'))},
    text: share.map((v) => `<b>${Math.round(100 * v)}%</b>`), textposition: 'outside', cliponaxis: false, textfont: {size: 11},
    hovertext: rs.map((r, k) => `activation ${r.lo.toFixed(3)}–${r.hi.toFixed(3)}<br><b>${Math.round(100 * share[k])}% NEI</b>: ` +
                                 `${fmt(r.nei)} of ${fmt(r.n)} claims`),
    hoverinfo: 'text',
  }]
  const layout = {
    xaxis: {title: {text: `how strongly the feature fires (activation) →${off >= 1 ? `   ·   not shown: ${fmt(off)} claims where it does not fire` : ''}`,
            font: {size: 11}}, tickfont: {size: 10}, type: 'category'},
    yaxis: {title: {text: '% of claims that are NEI', font: {size: 11}}, range: [0, 112], ticksuffix: '%', tickvals: [0, 25, 50, 75, 100]},
    shapes: [{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 100 * base, y1: 100 * base, line: {dash: 'dot', color: '#888', width: 1}}],
    annotations: [
      {xref: 'paper', x: 0, y: 100 * base, xanchor: 'left', yanchor: 'bottom', showarrow: false,
       text: `whole dataset: ${Math.round(100 * base)}%`, font: {size: 10, color: '#777'}},
      {xref: 'paper', yref: 'paper', x: 0, y: 1.08, xanchor: 'left', yanchor: 'bottom', showarrow: false,
       text: mirror ? '<span style="color:#2a78d6">■</span> NEI at most half as frequent as in the dataset'
                    : '<span style="color:#e8772e">■</span> most claims are NEI (above 50%)', font: {size: 11, color: '#444'}}],
    margin: {l: 55, r: 10, b: 70, t: 30}, autosize: true, showlegend: false, bargap: 0.2,
    hoverlabel: {bgcolor: '#fff', bordercolor: '#999', font: {size: 12, color: '#111'}, align: 'left'},
  };
  return <Plot style={{width: '100%', height: '100%'}} data={traces} layout={layout} config={{displayModeBar: false}}/>
}

export default HistogramDisplay;
