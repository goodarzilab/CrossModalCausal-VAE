#!/usr/bin/env python
"""ROUND2 copy of code/reviewfix/rf_dump.py (only change: uses analysis/fig3_zero_shot/ko_invariant_refs/op_controls_r2.py with R2_CACHE/R2_TAGS;
reference checks only when R2_CHECK=1). Original doc:
Items 2/3/4: dump signed-clamp (clamp_zero) predictions for one model seed under the pseudobulk DE (OP_DE=pb_bh),
by executing analysis/fig3_zero_shot/operator/op_controls.py unchanged (OP_PARTS=none) and calling its operator.
  (a) Fig 3 / Fig 4b,c sets, every vocabulary unit of each screen (so the joint bootstrap can score units that become
      evaluable in a replicate): learned A, A_rr transposed, A_rr off-diagonal shuffled (3 permutations, generator seeds
      0/1/2 exactly as code/dooperator/op_doop.py::direction_controls), A_rr removed (zeroed). Also delta_g (the clamp's
      on-target knockdown, z units) and the decoded control level, for the empirical-covariance comparator.
  (b) Fig 5 clamp|guard arms (baseline, real, shuffle_0..2, mean) re-fitted with the harness finetune_op and their
      predictions for TRAIN+TEST units saved (retrieval diagnosis, item 4); summary checked against op_pb_s_<seed>.json.
usage: python -I rf_dump.py RUN_DIR SEEDTAG OUTDIR
"""
import os as _os  # [public release] configurable paths (see docs/REPRODUCING.md)
_CMC_ROOT = _os.environ.get("CMCVAE_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../..")))
_CMC_DATA = _os.environ.get("CMCVAE_DATA", _os.path.join(_CMC_ROOT, "data"))
_CMC_RESULTS = _os.environ.get("CMCVAE_RESULTS", _os.path.join(_CMC_ROOT, "results"))
_CMC_SCRATCH = _os.environ.get("CMCVAE_SCRATCH", _os.path.join(_CMC_RESULTS, "scratch"))
_CMC_DEEPSEM = _os.environ.get("DEEPSEM_DIR", _os.path.join(_CMC_ROOT, "external", "DeepSEM"))
_CMC_SCVI_PY = _os.environ.get("SCVI_PY", "python")
import sys as _sys  # module search fallbacks (appended, so explicit sys.path inserts below take precedence)
_sys.path.extend(_p for _p in (_os.path.join(_CMC_ROOT, "src"), _os.path.join(_CMC_ROOT, "analysis", "common"), _os.path.join(_CMC_ROOT, "data_prep", "pseudobulk_de"), _os.path.join(_CMC_ROOT, "analysis", "fig3_zero_shot", "stats"), _os.path.join(_CMC_ROOT, "analysis", "figures", "original")) if _p not in _sys.path)
import os, sys, json, runpy
import numpy as np
B = f"{_CMC_RESULTS}/paper"
RUN, SEEDTAG, OUTD = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(OUTD, exist_ok=True)
os.environ["OP_DE"] = "pb_bh"; os.environ["OP_PARTS"] = "none"
sys.argv = ["op_controls.py", RUN, f"{OUTD}/opmeta_{SEEDTAG}.json"]
G = runpy.run_path(f"{_CMC_ROOT}/analysis/fig3_zero_shot/ko_invariant_refs/op_controls_r2.py", run_name="__opc__")
TAGS_ = G["TAGS"]
torch = G["torch"]; CTX, RAW, TAGS, DEVICE, lr_ = G["CTX"], G["RAW"], G["TAGS"], G["DEVICE"], G["lr_"]
G["model"].scm.load_state_dict(G["SCM0"])
A0 = G["assemble"]().detach().clone()
def put_rr(A, rr): A = A.clone(); A[:lr_, :lr_] = rr; return A
def shuf_off(M, s):
    M = M.clone(); n = M.shape[0]; off = ~torch.eye(n, dtype=torch.bool, device=M.device)
    v = M[off]; M[off] = v[torch.randperm(v.numel(), generator=torch.Generator().manual_seed(s)).to(M.device)]; return M
Arr = A0[:lr_, :lr_]
VARS = {"clamp": A0, "Arr_T": put_rr(A0, Arr.T.contiguous()), "Arr_zero": put_rr(A0, torch.zeros_like(Arr))}
for s in range(3): VARS[f"Arr_shuf{s}"] = put_rr(A0, shuf_off(Arr, s))
D = {}
SKIP3 = os.environ.get("R2_SKIP_FIG3") == "1"
with torch.no_grad():
    for t in ([] if SKIP3 else TAGS):
        al = RAW[t][0]; c = CTX[t]
        genes = sorted(g for g in RAW[t][1] if g in al["train_gene_idx"] and g in al["train_gene_to_idx_common"])
        gi = torch.as_tensor([al["train_gene_idx"][g] for g in genes], device=DEVICE)
        for nm, A in VARS.items():
            dr, dp, delta = G["predict_op"](A, G["ebar_of"](c), gi, "clamp", "zero")
            D[f"{nm}|{t}|genes"] = np.array(genes)
            D[f"{nm}|{t}|rna"] = dr.cpu().numpy()[:, c["reti"]].astype(np.float32)
            D[f"{nm}|{t}|prot"] = dp.cpu().numpy()[:, c["preti"]].astype(np.float32)
            if nm == "clamp":
                D[f"meta|{t}|delta"] = delta.cpu().numpy().astype(np.float64); D[f"meta|{t}|gidx"] = gi.cpu().numpy()
                D[f"meta|{t}|reti"] = np.asarray(c["reti"]); D[f"meta|{t}|preti"] = np.asarray(c["preti"])
        print(t, "units", len(genes), flush=True)
if not SKIP3: np.savez_compressed(f"{OUTD}/fig34_{SEEDTAG}.npz", **D)
# sanity: learned clamp must reproduce the operator run used for the NC Fig 3 (seed 42 file) / per-seed op_pb json
rows, pr, ont = G["fig3_rows"]("clamp_zero")
CHK = os.environ.get("R2_CHECK") == "1"
ref = {(r["cohort"], r["gene"]): r for r in json.load(open(f"{B}/analyses/operator/runs/op_pb_s_{SEEDTAG}.json"))["fig3"]["clamp_zero"]["per_ko"]} if CHK else {}
mx = {}
for r in rows:
    q = ref.get((r["cohort"], r["gene"]))
    if q is None: continue
    for m in ["rr_auroc", "rr_pe", "rr_pe_cent", "rr_retrieval", "rp_auroc", "rp_pe"]:
        a, b = r.get(m), q.get(m)
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)): continue
        mx[m] = max(mx.get(m, 0.0), abs(a - b))
chk = dict(fig3_max_abs_diff_vs_op_pb=mx, n_rows=len(rows), n_ref=len(ref))
print("CHECK fig3", chk, flush=True)

# ---- (b) Fig 5 clamp|guard arms with saved predictions
TRAIN, TEST = G["TRAIN"], G["TEST"]; TAGS = TAGS_
SPJ = os.environ.get("R2_SPLIT_JSON")
if SPJ:   # fresh gene split from code/benchmark/cite_bench.py (BM_SPLIT=k) run json: RES["split"]["train"/"test"]
    _sp = json.load(open(SPJ))["split"]
    TRAIN = {t: [g for g in _sp["train"].get(t, []) if g in CTX[t]["INFO"]] for t in TAGS}
    TEST = {t: [g for g in _sp["test"].get(t, []) if g in CTX[t]["INFO"]] for t in TAGS}
    G["set_S"](CTX, TRAIN); print("split override", SPJ, {t: (len(TRAIN[t]), len(TEST[t])) for t in TAGS}, flush=True)
ref5 = json.load(open(f"{B}/analyses/operator/runs/op_pb_s_{SEEDTAG}.json"))["fig5"]["clamp_zero|guard"] if CHK else None
P5 = {}; chk5 = {}
def dump_arm(name):
    pt, ont = G["predict_units_op"](CTX, {t: TRAIN[t] + TEST[t] for t in TAGS}, "clamp_zero")
    for t in TAGS:
        gl = TRAIN[t] + TEST[t]
        P5[f"{name}|{t}|genes"] = np.array(gl); P5[f"{name}|{t}|is_test"] = np.array([g in set(TEST[t]) for g in gl])
        P5[f"{name}|{t}|rna"] = np.stack([pt[(t, g)][0] for g in gl]).astype(np.float32)
        P5[f"{name}|{t}|prot"] = np.stack([pt[(t, g)][1] for g in gl]).astype(np.float32)
    rows = G["score"](CTX, TEST, pt, G["ref_means"](CTX, TRAIN, pt)); s = G["summarize"](rows)["overall"]
    chk5[name] = {m: [s[m], ref5[name]["summary"]["overall"][m] if ref5 else None] for m in ["rr_retrieval", "rr_pe", "rr_pe_cent", "rr_auroc"]}
    print("CHECK fig5", name, chk5[name], flush=True)
G["model"].scm.load_state_dict(G["SCM0"]); dump_arm("baseline")
G["finetune_op"](CTX, TRAIN, "clamp_zero"); dump_arm("real")
for k in range(3): G["finetune_op"](CTX, TRAIN, "clamp_zero", target="shuffle", perm_seed=k); dump_arm(f"shuffle_{k}")
G["finetune_op"](CTX, TRAIN, "clamp_zero", target="mean"); dump_arm("mean")
np.savez_compressed(f"{OUTD}/fig5arms_{SEEDTAG}.npz", **P5)
chk["fig5_vs_op_pb"] = chk5
json.dump(chk, open(f"{OUTD}/dump_check_{SEEDTAG}.json", "w"), indent=1, default=float)
print("DONE", flush=True)
