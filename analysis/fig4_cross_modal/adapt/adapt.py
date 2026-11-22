"""driver: python -I adapt.py BASE SEEDKEY DOMAINS VARIANTS   e.g. sp s42 frangieh,iccite,arce i,ii,iii,i_xm1
For each domain: diagnostics of the unadapted model ('none'), then each variant adapted from the base checkpoint, saved to
OUT/models/<base>_<variant>_<domain>_<seed>/ (config.json + final_model.pt, loadable by every harness), diagnostics to
OUT/diag/<base>_<variant>_<domain>_<seed>.json."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapt_lib import *
BASE, SK, DOMS, VARS = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4].split(",")
rd0 = BASES[BASE][SK]
for dom in DOMS:
    D = domain_data(dom); log(f"domain {dom}: sources {[(s['name'], len(s['tr']), len(s['va'])) for s in D['sources']]}")
    if dom in CITE:
        rg = ridge_screen(dom, D); log(f"ctrl-cell ridge held-out per-protein r median {rg['median']:.3f}")
        f = f"{OUT}/diag/{BASE}_none_{dom}_{SK}.json"
        if not os.path.exists(f):
            m, cfg = load_model(rd0); d = diag_screen(m, D); json.dump(dict(diag=d, ridge_median=rg["median"]), open(f, "w"))
            log(f"none {dom}: rna_only {d['rna_only']:.3f} with_prot {d['with_protein']:.3f} share {d['var_share_rna_path']:.3f}")
    for v in VARS:
        rd = model_dir(BASE, v, dom, SK)
        if os.path.exists(f"{rd}/final_model.pt"): log("exists", rd); continue
        m, cfg = load_model(rd0)
        info = run_adapt(m, D, v, SEEDNUM[SK])
        rec = dict(base=BASE, base_rd=rd0, variant=v, domain=dom, seed=SK, hp=HP, variant_def=VARIANTS[v],
                   best_step=info["best_step"], crit0=info["crit0"], crit_best=info["crit_best"], n_steps_run=info["n_steps_run"])
        out = dict(rec, hist=info["hist"])
        if dom in CITE:
            d = diag_screen(m, D); out["diag"] = d; out["ridge_median"] = rg["median"]
            log(f"{v} {dom}: best_step {info['best_step']} crit {info['crit0']:.2f}->{info['crit_best']:.2f} | rna_only {d['rna_only']:.3f} "
                f"with_prot {d['with_protein']:.3f} share {d['var_share_rna_path']:.3f} rna_rec {d['rna_recon_r_median']:.3f} (ridge {rg['median']:.3f})")
        else:
            log(f"{v} {dom}: best_step {info['best_step']} crit {info['crit0']:.2f}->{info['crit_best']:.2f} parts {info['hist'][0]['parts']} -> best")
        save_model(m, cfg, rd, rec)
        json.dump(out, open(f"{OUT}/diag/{BASE}_{v}_{dom}_{SK}.json", "w"), default=float)
log("DONE")
