import React, { useEffect, useState, useRef } from "react"
import { Link } from "react-router-dom"
import { normalizeSequences, Feature, FeatureInfo } from "../types"
import TokenHeatmap from "./tokenHeatmap";
import Histogram from "./histogram"
import Tooltip from "./tooltip"
import { get_feature_info } from "../interpAPI"
import { pathForFeature } from "../autoencoder_registry"

const LABEL_COLORS: {[key: string]: string} = {NEI: '#f0883e', SUP: '#2a9d5c', REF: '#d9534f', CONF: '#7a5cc4'}

const LabelBadge = ({label}: {label?: string}) => label ? (
  <span style={{background: LABEL_COLORS[label] || '#999', color: 'white', borderRadius: '3px', padding: '1px 6px',
                fontWeight: 'bold', fontSize: '11px'}}>{label}</span>
) : null

// FEVER keeps PTB bracket tokens in the evidence text; show them as plain punctuation.
const PTB: {[key: string]: string} = {'-LRB-': '(', '-RRB-': ')', '-LSB-': '[', '-RSB-': ']', '-LCB-': '{', '-RCB-': '}', '-COLON-': ':'}
const cleanEvidence = (e: string) => e.replace(/-(LRB|RRB|LSB|RSB|LCB|RCB|COLON)-/g, (m) => PTB[m])
const VERDICT_MARKER = ' ▶ verdict'

const pct = (x: number | null | undefined) => (x === null || x === undefined) ? 'n/a' : `${(100 * x).toFixed(1)}%`

export default ({ feature }: {feature: Feature}) => {
  const [data, setData] = useState(null as FeatureInfo | null)
  const [showingMore, setShowingMore] = useState({} as {[key: string]: boolean})
  const [showEvidence, setShowEvidence] = useState(true)
  const [isLoading, setIsLoading] = useState(true)
  const [got_error, setError] = useState(null as string | null)
  const currentFeatureRef = useRef(feature);

  useEffect(() => {
    async function fetchData() {
      setIsLoading(true)
      try {
        currentFeatureRef.current = feature;
        const result = await get_feature_info(feature)
        if (currentFeatureRef.current !== feature) return;
        // the data files mark the decision position with a Portuguese placeholder token; display it in English
        for (const s of [...(result.top || []), ...(result.random || [])]) {
          s.tokens = s.tokens.map((t: string) => t.trim() === '⟨decisão⟩' ? VERDICT_MARKER : t)
        }
        normalizeSequences(result.top, result.random)
        result.top.sort((a, b) => b.act - a.act);
        setData(result); setIsLoading(false); setError(null);
      } catch (e) {
        setError(String(e));
      }
    }
    fetchData()
  }, [feature])

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="w-8 h-8 border-4 border-gray-300 rounded-full animate-spin"></div>
        <div className="ml-2">loading feature {feature.atom}…</div>
        {got_error ? <span style={{color: 'red', marginLeft: 10}}>Error loading data: {got_error}</span> : null}
      </div>
    )
  }
  if (!data) throw new Error('no data.  this should not happen.')
  if (data.dead) return <div className="p-4">Feature {feature.atom} never fires at the <b>decision position</b> of the verdict prompts
    (it may still fire on other tokens of the prompt; statistics here cover the decision position only).</div>

  const fc = data.fc
  const all_sequences = [
    {label: 'Top activations', sequences: data.top, default_show: 10,
     note: 'The claims on which this feature fires most strongly at the verdict position.'},
    {label: 'Random positive activations', sequences: data.random, default_show: 5,
     note: 'A random sample of claims on which the feature fires at all (activation > 0): shows its typical values and labels.'},
  ]

  return (
    <div>
      {got_error ? <span style={{color: '#ee5555'}}>{got_error}</span> : null}
      <div className="text-sm text-gray-700 mb-3" style={{maxWidth: 900}}>
        <p className="mb-1"><b>How to read this page.</b> For every claim (from FEVER or VitaminC) the model reads a verdict prompt with
          the claim, its evidence and the instructions, and then writes its verdict. The feature is measured at a single point:
          the <b>last prompt token</b>, right before the model answers.</p>
        <ul style={{listStyle: 'disc', paddingLeft: 20}}>
          <li><span style={{border: '2px solid gray', borderRadius: 2, background: 'rgba(80, 220, 80, 0.5)'}}>{VERDICT_MARKER}</span> is
            not a word of the claim: it stands for that last token. Its shade of green is the feature's activation there
            (the claim tokens are left uncolored because only the verdict position is recorded).</li>
          <li><b>Gold label</b> is the dataset's correct answer, not the model's output. NEI = the evidence given is <i>not enough</i> to decide.</li>
          <li>Whether a claim is NEI depends on its <b>evidence</b>, not on the claim alone, so the evidence is shown under each claim.
            The same claim can appear more than once with different evidence.</li>
        </ul>
      </div>
      <div style={{flexDirection: 'row', display: 'flex', flexWrap: 'wrap'}}>
        <div style={{width: '520px', height: '220px'}}>
          <Histogram data={data} />
        </div>
        <table style={{marginLeft: '20px'}} className="text-sm">
          <tbody>
            <tr><td><Tooltip content={'Fraction active'} tooltip={'P(a > 0) over all decision positions'}/></td><td>{pct(fc?.frac_active)}</td></tr>
            <tr><td><Tooltip content={'NEI rate when active'} tooltip={'fraction of NEI claims among decisions where the feature fires'}/></td>
                <td>{pct(fc?.nei_rate_when_active)} <span className="text-gray-500">(base {pct(fc?.nei_base)})</span></td></tr>
            <tr><td><Tooltip content={'AUC NEI vs. SUP/REF'} tooltip={'how well the activation separates NEI from decided claims (0.5 = chance); computed from 40-bin histograms'}/></td>
                <td>{fc?.auc_nei_vs_dec_approx !== null && fc?.auc_nei_vs_dec_approx !== undefined ? fc.auc_nei_vs_dec_approx.toFixed(3) : 'n/a'}</td></tr>
            <tr><td><Tooltip content={'NEI in top-10 / top-20'} tooltip={'fraction of NEI claims among the claims that activate the feature most'}/></td>
                <td>{pct(fc?.nei_top10)} / {pct(fc?.nei_top20)}</td></tr>
            <tr><td><Tooltip content={'Mean'} tooltip={'E[a]'}/></td><td>{data.mean_act.toExponential(2)}</td></tr>
            <tr><td><Tooltip content={'E[a²]'} tooltip={'second moment'}/></td><td>{data.mean_act_squared.toExponential(2)}</td></tr>
          </tbody>
        </table>
        {data.logits &&
          <table style={{marginLeft: '20px'}} className="text-sm">
            <thead><tr><th colSpan={2}><Tooltip content={'Logit effects'} tooltip={'tokens most promoted / suppressed by the decoder direction (logit lens: final norm + unembedding)'}/></th></tr></thead>
            <tbody>
              {data.logits.up.map(([tok, v], i) => (
                <tr key={i}>
                  <td style={{color: '#2a9d5c'}}><code>{JSON.stringify(tok)}</code> {v.toFixed(2)}</td>
                  <td style={{color: '#d9534f', paddingLeft: 12}}><code>{JSON.stringify(data.logits!.down[i][0])}</code> {data.logits!.down[i][1].toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        }
        {data.neighbors &&
          <table style={{marginLeft: '20px'}} className="text-sm">
            <thead><tr><th><Tooltip content={'Nearest features'} tooltip={'cosine similarity between decoder directions'}/></th></tr></thead>
            <tbody>
              {data.neighbors.map(({atom, cos}) => (
                <tr key={atom}><td>
                  <Link className="text-blue-500 hover:text-blue-700" to={pathForFeature({...feature, atom})}>#{atom}</Link> {cos.toFixed(3)}
                </td></tr>
              ))}
            </tbody>
          </table>
        }
      </div>
      {all_sequences.map(({label, sequences, default_show, note}, idx) => {
        const n_show = showingMore[label] ? sequences.length : default_show;
        return (
          <React.Fragment key={idx}>
            <h3 className="text-md font-bold">
              {label}
              <button className="ml-2 mb-2 mt-2 text-sm text-gray-500"
                onClick={() => setShowingMore({...showingMore, [label]: !showingMore[label]})}>
                {showingMore[label] ? 'show less' : 'show more'}
              </button>
              <button className="ml-2 mb-2 mt-2 text-sm text-gray-500" onClick={() => setShowEvidence(!showEvidence)}>
                {showEvidence ? 'hide evidence' : 'show evidence'}
              </button>
            </h3>
            <p className="text-sm text-gray-600 mb-1">{note}</p>
            <table style={{fontSize: '12px'}} className="activations-table">
              <thead><tr><th>Claim #</th><th>Gold label</th><th>Source</th><th>Activation at verdict</th><th>Claim{showEvidence ? ' + evidence' : ''}</th></tr></thead>
              <tbody>
                {sequences.slice(0, n_show).map((sequence, i) => (
                  <tr key={i}>
                    <td className="center">{sequence.doc_id}</td>
                    <td className="center"><LabelBadge label={sequence.label}/></td>
                    <td className="center">{sequence.source}</td>
                    <td className="center">{sequence.act.toFixed(3)}</td>
                    <td className="p-2">
                      <TokenHeatmap info={sequence} renderNewlines={false}/>
                      {showEvidence && sequence.evidence ? <div className="text-gray-500 mt-1"><i>evidence:</i> {cleanEvidence(sequence.evidence)}</div> : null}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </React.Fragment>
        )
      })}
    </div>
  )
}
