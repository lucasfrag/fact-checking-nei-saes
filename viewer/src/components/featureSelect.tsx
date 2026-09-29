import React, { useState, FormEvent } from "react"
import { Feature } from "../types"
import { AUTOENCODER_FAMILIES, MODELS } from "../autoencoder_registry"
import { get_index } from "../interpAPI"

type FeatureSelectProps = {
  init_feature: Feature,
  onFeatureChange?: (feature: Feature) => void,
  onFeatureSubmit: (feature: Feature) => void,
  show_go?: boolean,
}

export default function FeatureSelect({init_feature, onFeatureChange, onFeatureSubmit, show_go}: FeatureSelectProps) {
  let [feature, setFeature] = useState(init_feature);
  const family = AUTOENCODER_FAMILIES[feature.autoencoder.family];
  const changeFeature = (f: Feature) => { onFeatureChange && onFeatureChange(f); setFeature(f); }
  // Random feature among those that fire at the decision position (listed in index.json).
  const feelingLuckySubmit = async () => {
    let atom = Math.floor(Math.random() * feature.autoencoder.num_features);
    try {
      const idx = await get_index(feature.autoencoder.path)
      atom = idx.features[Math.floor(Math.random() * idx.features.length)].atom
    } catch (e) { /* fall back to a uniform draw */ }
    const random_feature: Feature = {atom, autoencoder: feature.autoencoder}
    setFeature(random_feature); onFeatureSubmit(random_feature);
    return false
  }
  return (
    <h3>
      <b>Model</b>{" "}
      <select value={feature.autoencoder.H.model} onChange={(e) => changeFeature({
          atom: 0, autoencoder: family.get_ae({...feature.autoencoder.H, model: e.target.value})})}
        className="border border-gray-300 rounded-md p-1">
        {Object.entries(MODELS).map(([key, m]) => <option key={key} value={key}>{m.label}</option>)}
      </select>
      <span className="text-sm text-gray-500 ml-2">{family.label}</span>
      <br/>
      <b>Feature</b>
      <input type="number" value={feature.atom} min={0} max={feature.autoencoder.num_features - 1}
        style={{ width: 150, marginLeft: 10, marginRight: 10 }}
        onChange={(e) => (!isNaN(parseInt(e.target.value))) && changeFeature({...feature, atom: parseInt(e.target.value)})}
        onKeyDown={(e) => { if (e.key === 'Enter') onFeatureSubmit(feature) }}
        className="border border-gray-300 rounded-md p-2"/>
      {show_go &&
        <button onClick={(e: FormEvent) => { e.preventDefault(); onFeatureSubmit(feature); return false }}
          className="border border-gray-300 rounded-md p-2" style={{ width: 200 }}>
          Go to feature {feature.atom}
        </button>}
      <button onClick={feelingLuckySubmit} className="border border-gray-300 rounded-md p-2 ml-2">
        Random feature
      </button>
      <br/>
      <span className="text-xs text-gray-500">Claims and evidence come from VitaminC and FEVER (Wikipedia-derived, CC BY-SA 3.0).</span>
    </h3>
  )
}
