import React, { useEffect, useState, useRef } from "react"
import { Link } from "react-router-dom"
import { normalizeSequences, Feature, FeatureInfo, SteeringInfo } from "../types"
import TokenHeatmap from "./tokenHeatmap";
import Histogram, { bandSummary } from "./histogram"
import ClassDistributions from "./classDistributions"
import Tooltip from "./tooltip"
import { get_feature_info, get_steering } from "../interpAPI"
import { pathForFeature, STEERED } from "../autoencoder_registry"
import SteeringPanel from "./steeringPanel"

const LABEL_COLORS: {[key: string]: string} = {NEI: '#f0883e', SUP: '#2a9d5c', REF: '#d9534f', CONF: '#7a5cc4'}

const LabelBadge = ({label}: {label?: string}) => label ? (
  <span style={{background: LABEL_COLORS[label] || '#999', color: 'white', borderRadius: '3px', padding: '1px 6px',
                fontWeight: 'bold', fontSize: '11px'}}>{label}</span>
) : null

// FEVER keeps PTB bracket tokens in the evidence text; show them as plain punctuation.
const PTB: {[key: string]: string} = {'-LRB-': '(', '-RRB-': ')', '-LSB-': '[', '-RSB-': ']', '-LCB-': '{', '-RCB-': '}', '-COLON-': ':'}
const cleanEvidence = (e: string) => e.replace(/-(LRB|RRB|LSB|RSB|LCB|RCB|COLON)-/g, (m) => PTB[m])
const VERDICT_MARKER = ' ▶ verdict'

// one plain-language sentence from the stats (no claims beyond what the numbers show)
const summary = (fc: NonNullable<FeatureInfo['fc']>) => {
  const auc = fc.auc_nei_vs_dec_approx
  const where = `It fires on ${(100 * fc.frac_active).toFixed(fc.frac_active < 0.01 ? 2 : 0)}% of verdicts. `
  if (auc === null || auc === undefined) return where
  if (auc >= 0.6) return where + `It tends to be stronger when the evidence is NOT enough to decide: ${Math.round(100 * (fc.nei_top20 ?? 0))}% of its 20 strongest claims are NEI, against ${Math.round(100 * fc.nei_base)}% on average.`
  if (auc <= 0.4) return where + `It tends to be stronger when the claim CAN be decided (supported or refuted): only ${Math.round(100 * (fc.nei_top20 ?? 0))}% of its 20 strongest claims are NEI, against ${Math.round(100 * fc.nei_base)}% on average.`
  return where + `Its strength is not clearly related to whether the evidence is enough (separation score close to 0.5).`
}

const pct = (x: number | null | undefined) => (x === null || x === undefined) ? 'n/a' : `${(100 * x).toFixed(1)}%`

export default ({ feature }: {feature: Feature}) => {
  const [data, setData] = useState(null as FeatureInfo | null)
  const [steering, setSteering] = useState(null as SteeringInfo | null)
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
    setSteering(null)
    if ((STEERED[feature.autoencoder.path] || []).includes(feature.atom)) {
      get_steering(feature.autoencoder.path, feature.atom)
        .then((s) => { if (currentFeatureRef.current === feature) setSteering(s) }).catch(() => {})
    }
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
      <details className="text-sm text-gray-700 mb-3" style={{maxWidth: 900}}>
        <summary style={{cursor: 'pointer'}}><b>How to read this page</b></summary>
        <p className="mb-1 mt-1">For every claim (from FEVER or VitaminC) the model reads a verdict prompt with
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
      </details>
      <div style={{maxWidth: 950}}>
        <h3 className="text-md font-bold">How strongly does this feature fire on NEI claims vs SUP/REF claims?</h3>
        {fc && (() => {
          const hb = data.hist_by_label
          if (!hb) return null
          const m = (c: number[]) => { const t = c.reduce((a, b) => a + b, 0) || 1
            return c.reduce((a, v, i) => a + v * (hb.edges[i] + hb.edges[i + 1]) / 2, 0) / t }
          const mN = m(hb['NEI'] || []), mD = m(hb['SUP/REF'] || []), r = mN / (mD || 1)
          return <p className="text-sm mb-1">Each curve shows how the activations of one group of claims are spread.{' '}
            {r >= 1 ? <>The <b style={{color: '#e8772e'}}>orange</b> curve is further to the right: this feature fires <b>more strongly when the
              evidence is not enough</b> (on average {r.toFixed(1)}× as strongly as on SUP/REF claims).</>
             : <>The <b style={{color: '#2a78d6'}}>blue</b> curve is further to the right: this feature fires <b>more strongly when the evidence
              is enough</b> to decide (on average {(1 / r).toFixed(1)}× as strongly as on NEI claims).</>}</p>
        })()}
        {fc && <div style={{height: '300px', maxWidth: 950}}><ClassDistributions data={data}/></div>}
        <details className="mt-2">
        <summary className="text-sm" style={{cursor: 'pointer'}}>More detail: share of NEI by how strongly the feature fires</summary>
        <p className="text-sm mb-1 mt-2"><b>{bandSummary(data)}</b></p>
        {fc &&
          <div className="text-sm text-gray-700 mb-1" style={{display: 'flex', gap: 18, flexWrap: 'wrap'}}>
            <Tooltip content={<span>fires on <b>{pct(fc.frac_active)}</b> of answers</span>} tooltip={'share of all verdicts where the feature is active (> 0)'}/>
            <Tooltip content={<span>NEI among its 20 strongest: <b>{pct(fc.nei_top20)}</b></span>} tooltip={'of the 20 claims that activate it most, how many have NEI as the correct answer (average 20%)'}/>
            <Tooltip content={<span>AUC <b>{fc.auc_nei_vs_dec_approx !== null && fc.auc_nei_vs_dec_approx !== undefined ? fc.auc_nei_vs_dec_approx.toFixed(3) : 'n/a'}</b></span>}
              tooltip={'chance that a random NEI claim activates it more than a random SUP/REF claim (0.5 = no link; below 0.5 = fires more on SUP/REF claims)'}/>
          </div>}
        <div style={{height: '320px'}}><Histogram data={data} /></div>
        </details>
      </div>
      {(data.logits || data.neighbors) &&
        <details className="mt-2 mb-3 text-sm">
          <summary style={{cursor: 'pointer'}} className="text-gray-700">Technical details (for researchers)</summary>
          <div style={{display: 'flex', flexWrap: 'wrap', gap: 30, marginTop: 8}}>
            {data.logits &&
              <div style={{maxWidth: 420}}>
                <b>Logit effects</b>
                <p className="text-gray-600 mb-1">If this feature pushed the model directly towards its output, which tokens would become more
                  (green) or less (red) likely? (logit lens: decoder direction → final norm → unembedding). Meaningless-looking tokens
                  mean the feature does not act on specific words directly, but through later layers.</p>
                <table><tbody>
                  {data.logits.up.map(([tok, v], i) => (
                    <tr key={i}>
                      <td style={{color: '#2a9d5c'}}><code>{JSON.stringify(tok)}</code> {v.toFixed(2)}</td>
                      <td style={{color: '#d9534f', paddingLeft: 12}}><code>{JSON.stringify(data.logits!.down[i][0])}</code> {data.logits!.down[i][1].toFixed(2)}</td>
                    </tr>))}
                </tbody></table>
              </div>}
            {data.neighbors &&
              <div style={{maxWidth: 300}}>
                <b>Nearest features</b>
                <p className="text-gray-600 mb-1">Features whose direction inside the model is most similar to this one (cosine similarity
                  of decoder directions; 1 = identical). High values would mean the concept is split across several features.</p>
                <table><tbody>
                  {data.neighbors.map(({atom, cos}) => (
                    <tr key={atom}><td>
                      <Link className="text-blue-500 hover:text-blue-700" to={pathForFeature({...feature, atom})}>#{atom}</Link> {cos.toFixed(3)}
                    </td></tr>))}
                </tbody></table>
              </div>}
          </div>
        </details>}
      {steering ? <SteeringPanel data={steering} path={feature.autoencoder.path}/> : null}
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
