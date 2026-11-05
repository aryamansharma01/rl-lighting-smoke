"""T3: build self-contained labeling pages (one per annotator) + a private key file.

Selection (default: ALL 192 images of the 40-step T1 set, i.e. every prompt x seed):
  * labeling every seed makes best-of-8 vs random (T5) exact and means the pages do not depend
    on verifier output, so labeling can start right after T1. With --total N < all, seeds are
    drawn per prompt (stratified); if --scores is given the verifier's top-1 is always kept.
  * 30 images are SHARED (labeled by all three annotators, for agreement); the rest are split
    evenly between annotators, stratified so each annotator sees every prompt.
  With 192 images: 30 shared + 54 own = 84 images per annotator (~25 min).

Annotators never see prompt, target, or seed: images get opaque ids, and the mapping lives in
label/key.csv (don't open it while labeling).

Usage:
  python make_label_pages.py --manifest ../results/T1/manifest_40steps.jsonl --root ../results/T1 \
      [--scores ../results/T2/scores_40steps.jsonl] [--total 150] --annotators A B C
Outputs: label/label_<annotator>.html, label/key.csv, label/selection.json
"""
import argparse
import base64
import csv
import hashlib
import io
import json
import random
from collections import defaultdict
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent

HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Light-direction labeling: __ANN__</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#1e1e1e;color:#eee;display:flex;flex-direction:column;align-items:center}
 header{width:100%;max-width:900px;padding:10px 16px;box-sizing:border-box}
 #img{max-width:min(90vw,640px);max-height:70vh;border:3px solid #444;image-rendering:auto}
 .keys{display:flex;gap:10px;margin:10px 0;flex-wrap:wrap;justify-content:center}
 .k{padding:8px 14px;border-radius:6px;background:#333;cursor:pointer;user-select:none;border:2px solid transparent}
 .k.sel{border-color:#4caf50;background:#2e4d30}
 #status{margin:6px;color:#bbb} button{padding:8px 14px;margin:4px;cursor:pointer}
 details{max-width:820px;color:#ccc;font-size:14px} .done{color:#4caf50}
</style></head><body>
<header>
 <b>Annotator __ANN__</b> &middot; <span id="prog"></span>
 <details open><summary>Instructions (press H to toggle)</summary>
 <p>For each image, decide where the light comes from, <b>judged from the main object's cast shadow</b>, in image coordinates.</p>
 <ul>
  <li><b>L</b>: light comes from the <b>left</b> side of the image (the shadow falls toward the right).</li>
  <li><b>R</b>: light comes from the <b>right</b> side of the image (the shadow falls toward the left).</li>
  <li><b>A</b>: a shadow is visible but the direction is ambiguous or not left/right (straight toward or away from the camera, shadows that contradict each other, can't tell).</li>
  <li><b>N</b>: no visible cast shadow of the main object.</li>
 </ul>
 <p>A diagonal shadow (e.g. falling to the back-right) counts as the side it leans toward: back-right means <b>L</b>. Keys: L / R / A / N to label and advance &middot; &larr; / &rarr; to move &middot; Backspace to go back. Progress autosaves in this browser. Click <b>Export CSV</b> when done and send the file.</p>
 </details>
</header>
<img id="img" alt="">
<div class="keys">
 <span class="k" data-l="L">L &middot; from left</span><span class="k" data-l="R">R &middot; from right</span>
 <span class="k" data-l="A">A &middot; ambiguous/other</span><span class="k" data-l="N">N &middot; no shadow</span>
</div>
<div id="status"></div>
<div><button id="exp">Export CSV</button><button id="next_un">Jump to next unlabeled</button></div>
<script>
const ANN="__ANN__"; const ITEMS=__ITEMS__; const KEY="lightlabels_v1_"+ANN;
let labels=JSON.parse(localStorage.getItem(KEY)||"{}"); let i=0;
const img=document.getElementById("img");
function save(){localStorage.setItem(KEY,JSON.stringify(labels));}
function n_done(){return ITEMS.filter(x=>labels[x.id]).length;}
function show(){
  const it=ITEMS[i]; img.src=it.src;
  document.querySelectorAll(".k").forEach(k=>k.classList.toggle("sel",labels[it.id]&&labels[it.id].label===k.dataset.l));
  const d=n_done();
  document.getElementById("prog").innerHTML=`image ${i+1} / ${ITEMS.length} &middot; labeled ${d}`+(d===ITEMS.length?' <span class="done">(all done, export!)</span>':"");
  document.getElementById("status").textContent=labels[it.id]?("current label: "+labels[it.id].label):"unlabeled";
}
function setLabel(l){const it=ITEMS[i]; labels[it.id]={label:l,t:new Date().toISOString()}; save();
  if(i<ITEMS.length-1) i++; show();}
function exportCSV(){
  let rows=["annotator,item_id,label,timestamp"];
  ITEMS.forEach(x=>{const v=labels[x.id]; rows.push([ANN,x.id,v?v.label:"",v?v.t:""].join(","));});
  const a=document.createElement("a"); a.href=URL.createObjectURL(new Blob([rows.join("\n")+"\n"],{type:"text/csv"}));
  a.download="labels_"+ANN+".csv"; a.click();
  if(n_done()<ITEMS.length) alert((ITEMS.length-n_done())+" images are still unlabeled (exported anyway).");
}
document.addEventListener("keydown",e=>{
  const k=e.key.toUpperCase();
  if(["L","R","A","N"].includes(k)) setLabel(k);
  else if(e.key==="ArrowRight"&&i<ITEMS.length-1){i++;show();}
  else if((e.key==="ArrowLeft"||e.key==="Backspace")&&i>0){i--;show();e.preventDefault();}
  else if(k==="H"){const d=document.querySelector("details"); d.open=!d.open;}
});
document.querySelectorAll(".k").forEach(k=>k.onclick=()=>setLabel(k.dataset.l));
document.getElementById("exp").onclick=exportCSV;
document.getElementById("next_un").onclick=()=>{const j=ITEMS.findIndex(x=>!labels[x.id]); if(j>=0){i=j;show();}};
const first=ITEMS.findIndex(x=>!labels[x.id]); i=first>=0?first:0; show();
</script></body></html>
"""


def opaque_id(image_id, salt):
    return "img_" + hashlib.sha1((salt + image_id).encode()).hexdigest()[:10]


def data_uri(path, size=512):
    im = Image.open(path).convert("RGB")
    if max(im.size) > size:
        im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True, help="dir the manifest image paths are relative to")
    ap.add_argument("--scores", help="verifier scores jsonl (non-flipped rows used for top-1)")
    ap.add_argument("--annotators", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--total", type=int, default=0, help="0 = label every image (default)")
    ap.add_argument("--shared", type=int, default=None,
                    help="images labeled by all annotators (default 30; fast_mode.json can set label_shared)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=HERE, help="where pages, key.csv and selection.json go")
    args = ap.parse_args()
    sys.path.insert(0, str(HERE.parent / "env"))
    import fast_mode
    if args.shared is None:
        args.shared = fast_mode.cfg().get("label_shared", 30)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.manifest)]
    by_prompt = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_id"]].append(r)
    prompts = sorted(by_prompt)

    top1 = {}
    if args.scores:
        best = {}
        for l in open(args.scores):
            s = json.loads(l)
            if s["flipped"]:
                continue
            key = (s["reward"], -s["seed"])  # deterministic tie-break: lower seed wins
            if s["prompt_id"] not in best or key > best[s["prompt_id"]][0]:
                best[s["prompt_id"]] = (key, s["image_id"])
        top1 = {p: v[1] for p, v in best.items()}

    if args.total <= 0:
        args.total = len(rows)
    # how many per prompt: spread `total` as evenly as possible over prompts
    base, extra = divmod(args.total, len(prompts))
    order = prompts[:]
    rng.shuffle(order)
    quota = {p: base + (1 if j < extra else 0) for j, p in enumerate(order)}

    selected = []  # (row, is_top1)
    for p in prompts:
        items = sorted(by_prompt[p], key=lambda r: r["seed"])
        t = top1.get(p)
        chosen = [r for r in items if r["image_id"] == t]
        rest = [r for r in items if r["image_id"] != t]
        rng.shuffle(rest)
        chosen += rest[: quota[p] - len(chosen)]
        selected += [(r, r["image_id"] == t) for r in chosen]

    # shared set: stratified over prompts (round-robin over shuffled prompts)
    per_p = defaultdict(list)
    for r, _ in selected:
        per_p[r["prompt_id"]].append(r["image_id"])
    for v in per_p.values():
        rng.shuffle(v)
    shared, pp = [], prompts[:]
    rng.shuffle(pp)
    while len(shared) < args.shared:
        for p in pp:
            if per_p[p] and len(shared) < args.shared:
                shared.append(per_p[p].pop())
    shared_set = set(shared)

    # remaining images: deal round-robin per prompt so each annotator sees every prompt
    own = {a: [] for a in args.annotators}
    k = 0
    for p in pp:
        for iid in per_p[p]:
            own[args.annotators[k % len(args.annotators)]].append(iid)
            k += 1

    salt = f"lightlabel{args.seed}"
    meta = {r["image_id"]: r for r, _ in selected}
    is_top = {r["image_id"]: t for r, t in selected}
    with open(out_dir / "key.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "image_id", "prompt_id", "seed", "target_dir", "phrasing",
                    "object", "scene", "shared", "assigned_to", "verifier_top1"])
        for iid, r in sorted(meta.items()):
            who = "ALL" if iid in shared_set else next(a for a in own if iid in own[a])
            w.writerow([opaque_id(iid, salt), iid, r["prompt_id"], r["seed"], r["target_dir"],
                        r["phrasing"], r["object"], r["scene"], int(iid in shared_set), who,
                        int(is_top[iid])])

    uri_cache = {}
    for a in args.annotators:
        ids = shared + own[a]
        rng_a = random.Random(f"{args.seed}-{a}")
        rng_a.shuffle(ids)  # shared images interleaved, different order per annotator
        items = []
        for iid in ids:
            if iid not in uri_cache:
                uri_cache[iid] = data_uri(Path(args.root) / meta[iid]["image"])
            items.append(dict(id=opaque_id(iid, salt), src=uri_cache[iid]))
        html = HTML.replace("__ANN__", a).replace("__ITEMS__", json.dumps(items))
        (out_dir / f"label_{a}.html").write_text(html)
        print(f"annotator {a}: {len(items)} images -> label_{a}.html")

    json.dump(dict(total=len(selected), shared=shared, own=own, top1=top1, quota=quota, seed=args.seed),
              open(out_dir / "selection.json", "w"), indent=1)
    print(f"selected {len(selected)} images ({len(shared)} shared); key -> {out_dir / 'key.csv'}")


if __name__ == "__main__":
    main()
