import {Feature, FeatureInfo, FeatureIndex} from './types';

// Per-feature JSONs live in the companion Hugging Face dataset (CORS-enabled):
//   viewer/<model>/layer-<L>/atoms/<atom // 1000>/<atom>.json   and   viewer/<model>/layer-<L>/index.json
export const DATA_BASE = "https://huggingface.co/datasets/lucasfrag/fact-checking-abstention-saes-data/resolve/main/viewer"

const load_json = async (url: string) => {
  const res = await fetch(url, {method: "GET", mode: "cors"})
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} - ${res.statusText} (${url})`)
  }
  return await res.json()
}

const cache: {[key: string]: any} = {}

export async function get_feature_info(feature: Feature): Promise<FeatureInfo> {
  const ae = feature.autoencoder;
  const url = `${DATA_BASE}/${ae.path}/atoms/${Math.floor(feature.atom / 1000)}/${feature.atom}.json`
  if (!(url in cache)) cache[url] = await load_json(url)
  return cache[url]
}

export async function get_index(path: string): Promise<FeatureIndex> {
  const url = `${DATA_BASE}/${path}/index.json`
  if (!(url in cache)) cache[url] = await load_json(url)
  return cache[url]
}
