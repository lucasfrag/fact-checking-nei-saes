import React, { useState } from 'react'

// "Count the dots": each dot is one claim, colored by its correct (gold) label. Left: the 20 claims where the feature
// fires most strongly (the `top` list of the feature JSON, rank 1 first). Right: what 20 claims look like on average in
// the whole dataset (round(20 × NEI rate) NEI dots, spread out). Hover a dot on the left to read the claim.
const NEI = '#f0883e', OTHER = '#7fa7d9'
const N = 20, COLS = 5

type Top = {label?: string, tokens?: string[], evidence?: string, act?: number}

const Dot = ({nei, title, onEnter, onLeave}: {nei: boolean, title?: string, onEnter?: () => void, onLeave?: () => void}) => (
  <span title={title} onMouseEnter={onEnter} onMouseLeave={onLeave}
    style={{width: 22, height: 22, borderRadius: '50%', display: 'inline-block', background: nei ? NEI : OTHER,
            cursor: onEnter ? 'help' : 'default', boxShadow: '0 0 0 1px rgba(0,0,0,.08) inset'}}/>
)

const Grid = ({children}: {children: React.ReactNode}) => (
  <div style={{display: 'grid', gridTemplateColumns: `repeat(${COLS}, 22px)`, gap: 8}}>{children}</div>
)

export default function DotCompare({top, base}: {top: Top[], base: number}) {
  const [hover, setHover] = useState(null as number | null)
  const t = top.slice(0, N)
  if (t.length < N) return null
  const nTop = t.filter((x) => x.label === 'NEI').length
  const nBase = Math.round(N * base)
  // spread the expected NEI dots evenly over the 20 positions
  const baseNei = new Set(Array.from({length: nBase}, (_, i) => Math.floor((i + 0.5) * N / nBase)))
  const claim = (x: Top) => (x.tokens || []).slice(0, -1).join('').trim()
  const h = hover === null ? null : t[hover]

  return (
    <div>
      <div style={{display: 'flex', gap: 48, flexWrap: 'wrap', alignItems: 'flex-start', marginTop: 6}}>
        <div style={{display: 'grid', gap: 8}}>
          <div className="text-sm text-gray-700">The <b>20 claims</b> where this feature fires <b>most strongly</b></div>
          <Grid>
            {t.map((x, i) => <Dot key={i} nei={x.label === 'NEI'}
              onEnter={() => setHover(i)} onLeave={() => setHover(null)}/>)}
          </Grid>
          <div style={{fontSize: 22, fontWeight: 700}}>{nTop} of {N} <span style={{fontSize: 14, fontWeight: 400}}>are NEI</span></div>
        </div>
        <div style={{display: 'grid', gap: 8}}>
          <div className="text-sm text-gray-700"><b>20 claims</b> of the dataset, <b>on average</b></div>
          <Grid>
            {Array.from({length: N}, (_, i) => <Dot key={i} nei={baseNei.has(i)}/>)}
          </Grid>
          <div style={{fontSize: 22, fontWeight: 700}}>{nBase} of {N} <span style={{fontSize: 14, fontWeight: 400}}>are NEI</span></div>
        </div>
      </div>
      <div className="text-sm mt-2" style={{display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'center'}}>
        <span><Dot nei={true}/> <span style={{verticalAlign: 'top'}}>NEI: the evidence was <b>not enough</b> to decide</span></span>
        <span><Dot nei={false}/> <span style={{verticalAlign: 'top'}}>SUP or REF: the evidence was enough</span></span>
      </div>
      <div className="text-sm mt-2" style={{minHeight: 44, color: '#444', maxWidth: 900}}>
        {h ? <><b>#{(hover as number) + 1} strongest</b> · correct answer <b>{h.label}</b> · “{claim(h)}”
                {h.evidence ? <span className="text-gray-500"> — evidence: {h.evidence.slice(0, 160)}{h.evidence.length > 160 ? '…' : ''}</span> : null}</>
           : <span className="text-gray-500">Hover a dot on the left to read the claim.</span>}
      </div>
    </div>
  )
}
