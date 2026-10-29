# re-id evaluation on market-1501: the old imagenet mobilenetv3 vs osnet_x0_25 (msmt17),
# on head-and-shoulders crops and full person crops.
# python tools/eval_reid.py --market path/to/Market-1501-v15.09.15
#
# 1. standard protocol: rank-1 and mAP, query vs gallery (bounding_box_test)
# 2. the journal as the app uses it: n people stored with up to k vectors each; events are
#    repeat violators seen by another camera, or new people. the match threshold is picked on
#    bounding_box_train identities (<=5% of new people merged into a profile at n=50) and
#    reported on the test identities
# osnet weights download on first run; market-1501 has to be downloaded separately
import argparse
import glob
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reid import load_osnet, preprocess  # noqa: E402


def load_mobilenet():
    import torchvision.models as M
    b = M.mobilenet_v3_small(weights=M.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    return torch.nn.Sequential(b.features, b.avgpool, torch.nn.Flatten()).eval()


MODELS = {"mobilenetv3_imagenet": load_mobilenet, "osnet_x0_25_msmt17": load_osnet}
CROPS = {"head": 0.55, "full": 1.0}  # top part of the person box


def parse(paths):
    ids = [re.match(r"(-?\d+)_c(\d)", Path(p).name).groups() for p in paths]
    return np.array([int(a) for a, _ in ids]), np.array([int(b) for _, b in ids])


@torch.no_grad()
def embed(model, paths, top, cache: Path, bs=64):
    if cache.exists():
        return np.load(cache)
    import cv2
    out = []
    for i in range(0, len(paths), bs):
        batch = []
        for p in paths[i:i + bs]:
            im = cv2.imread(p)
            batch.append(preprocess(im[: int(im.shape[0] * top)]))
        f = model(torch.from_numpy(np.stack(batch)))
        out.append(torch.nn.functional.normalize(f, dim=1).numpy())
        print(f"\r  {cache.stem}: {min(i + bs, len(paths))}/{len(paths)}", end="", flush=True)
    print()
    feats = np.concatenate(out)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, feats)
    return feats


def market_protocol(qf, qp, qc, gf, gp, gc):
    sims = qf @ gf.T
    r1 = ap = n = 0.0
    for i in range(len(qf)):
        keep = ~((gp == qp[i]) & (gc == qc[i]))  # same id seen by the same camera doesn't count
        match = gp[keep][np.argsort(-sims[i][keep])] == qp[i]
        if not match.any():
            continue
        n += 1
        r1 += match[0]
        prec = np.cumsum(match) / (np.arange(len(match)) + 1)
        ap += (prec * match).sum() / match.sum()
    return r1 / n, ap / n


def journal(F, p, c, thr, n_known, k, trials=300, seed=0):
    rng = random.Random(seed)
    by: dict[int, list[int]] = {}
    for i, x in enumerate(p):
        by.setdefault(int(x), []).append(i)
    ids = sorted(x for x, v in by.items() if len(v) >= 2)
    ok = wrong = missed = new_wrong = new_total = 0
    for _ in range(trials):
        pick = rng.sample(ids, n_known + 20)
        known, new_ids = pick[:n_known], pick[n_known:]
        owners, vecs, held = [], [], {}
        for kid in known:
            imgs = by[kid][:]
            rng.shuffle(imgs)
            held[kid] = imgs[0]  # comes back later as the repeat event
            stored = [i for i in imgs[1:] if c[i] != c[imgs[0]]][:k] or imgs[1:2]
            owners += [kid] * len(stored)
            vecs += [F[i] for i in stored]
        D, owners = np.stack(vecs), np.array(owners)
        for kid in rng.sample(known, min(20, n_known)):
            s = D @ F[held[kid]]
            b = int(np.argmax(s))
            if s[b] < thr:
                missed += 1
            elif owners[b] == kid:
                ok += 1
            else:
                wrong += 1
        for kid in new_ids:
            new_total += 1
            new_wrong += (D @ F[rng.choice(by[kid])]).max() >= thr
    rep = ok + wrong + missed
    return {"repeat_found": round(ok / rep, 3), "repeat_merged_wrong": round(wrong / rep, 3),
            "repeat_new_profile": round(missed / rep, 3),
            "new_person_merged_wrong": round(new_wrong / new_total, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True, type=Path)
    ap.add_argument("--cache", type=Path, default=ROOT / "runs" / "reid_features")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "reid_eval.json")
    a = ap.parse_args()

    q = sorted(glob.glob(str(a.market / "query" / "*.jpg")))
    g = [x for x in sorted(glob.glob(str(a.market / "bounding_box_test" / "*.jpg")))
         if not Path(x).name.startswith("-1")]  # -1 = junk boxes
    tr = sorted(glob.glob(str(a.market / "bounding_box_train" / "*.jpg")))
    random.Random(1).shuffle(tr)
    tr = tr[:4000]
    (qp, qc), (gp, gc), (tp, tc) = parse(q), parse(g), parse(tr)
    test_keep = np.concatenate([qp, gp]) > 0  # id 0 = distractors, no repeat sightings

    res = {"queries": len(q), "gallery": len(g), "calibration_images": len(tr)}
    for name, load in MODELS.items():
        model = load()
        for crop, top in CROPS.items():
            key = f"{name}/{crop}"
            qf = embed(model, q, top, a.cache / f"{name}_{crop}_query.npy")
            gf = embed(model, g, top, a.cache / f"{name}_{crop}_gallery.npy")
            tf = embed(model, tr, top, a.cache / f"{name}_{crop}_train.npy")
            r1, mAP = market_protocol(qf, qp, qc, gf, gp, gc)
            res[key] = {"rank1": round(float(r1), 3), "mAP": round(float(mAP), 3)}
            F = np.concatenate([qf, gf])[test_keep]
            P, C = np.concatenate([qp, gp])[test_keep], np.concatenate([qc, gc])[test_keep]
            for k in (1, 3):
                thr = next(t / 1000 for t in range(500, 1000, 5)
                           if journal(tf, tp, tc, t / 1000, 50, k, trials=150)["new_person_merged_wrong"] <= 0.05)
                res[key][f"journal_k{k}"] = {"threshold": thr,
                                             **{f"N={n}": journal(F, P, C, thr, n, k) for n in (20, 50, 200)}}
            print(key, json.dumps(res[key]))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    print(f"saved: {a.out}")


if __name__ == "__main__":
    main()
