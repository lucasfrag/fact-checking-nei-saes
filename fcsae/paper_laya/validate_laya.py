"""Validação do Laya como juiz externo de suficiência da evidência (ideia para paper; LEIA-ME "IDEIA PARA PAPER").

Amostra estratificada dos pares (alegação, evidência) de treino do SAE (examples.jsonl; mesma ordem para todos os
modelos): n_nei NEI + n_dec/2 SUP + n_dec/2 REF, semente fixa. Para cada par, pergunta de escolha com chaves neutras
(recomendação do model card: `noul` pode seguir os rótulos das opções), no formato NLI em que o Laya é melhor (XNLI 0,86):
  A = a evidência apoia; B = contradiz; C = não basta para decidir  →  P(insuficiente) = P(C).
Roda só em CPU (CUDA_VISIBLE_DEVICES vazio). Grava uma linha JSON por par (retomável).

    USE_TF=0 CUDA_VISIBLE_DEVICES= ~/venvs/laya/bin/python fcsae/paper_laya/validate_laya.py --n-nei 10000 --n-dec 10000
"""
import argparse, json, os, random, time

EX = "/mnt/d/Projetos WSL/fcsae-store/activations/v3_llama/examples.jsonl"
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "laya_validation")
QUESTION = {"sufficiency": {
    "type": "choice",
    "instructions": "Does the evidence allow deciding whether the claim is true or false?",
    "criteria": {"A": "yes, the evidence supports the claim",
                 "B": "yes, the evidence contradicts the claim",
                 "C": "no, the evidence is not enough to decide"}}}


def sample(n_nei, n_dec, seed=0):
    by = {"SUP": [], "REF": [], "NEI": []}
    for i, l in enumerate(open(EX, encoding="utf-8")):
        lab = json.loads(l)["label"]
        if lab in by:
            by[lab].append(i)
    rng = random.Random(seed)
    idx = rng.sample(by["NEI"], n_nei) + rng.sample(by["SUP"], n_dec // 2) + rng.sample(by["REF"], n_dec - n_dec // 2)
    return sorted(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nei", type=int, default=10000); ap.add_argument("--n-dec", type=int, default=10000)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--threads", type=int, default=20)
    a = ap.parse_args()
    import torch, laya
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    idx = sample(a.n_nei, a.n_dec)
    if a.limit:
        idx = idx[:a.limit]
    path = os.path.join(OUT, f"preds_{a.n_nei}_{a.n_dec}{'_lim' + str(a.limit) if a.limit else ''}.jsonl")
    done = {json.loads(l)["idx"] for l in open(path)} if os.path.exists(path) else set()
    want = set(idx)
    exs = {}
    for i, l in enumerate(open(EX, encoding="utf-8")):
        if i in want and i not in done:
            exs[i] = json.loads(l)
    agent = laya.load("convaiinnovations/laya")
    t0 = time.time(); n = 0
    with open(path, "a", encoding="utf-8") as f:
        for i in sorted(exs):
            e = exs[i]
            state = {"claim": e["claim"], "evidence": "\n".join(f"[{k + 1}] {t}" for k, t in enumerate(e["evidence"][:4]))}
            r = agent.predict(state, QUESTION)["answers"]["sufficiency"]
            f.write(json.dumps({"idx": i, "label": e["label"], "source": e.get("source"), "answer": r}, ensure_ascii=False) + "\n")
            n += 1
            if n % 200 == 0:
                f.flush(); print(f"{n}/{len(exs)} | {n / (time.time() - t0):.2f}/s", flush=True)
    print(f"FIM: {len(done) + n} pares em {path}", flush=True)


if __name__ == "__main__":
    main()
