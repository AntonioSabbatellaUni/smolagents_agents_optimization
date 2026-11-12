"""Revival: convert one GAIA validation attachment per extension with MarkdownConverter (xlsx/pdf/docx/pptx/csv/zip/mp3/...),
print status + head of the text; also list subset tasks (MALBO_TASK_IDS_FILE) with their attachment type.
Run from this directory inside the container."""
import json, os, sys, time, glob, collections
import pandas as pd
from scripts.mdconvert import MarkdownConverter

val = "data/gaia/2023/validation"
meta = pd.read_parquet(f"{val}/metadata.parquet")
meta = meta[meta.file_name.fillna("") != ""]
if os.getenv("MALBO_TASK_IDS_FILE"):
    d = json.load(open(os.environ["MALBO_TASK_IDS_FILE"]))
    ids = {t["task_id"] for t in d["tasks"]}
    print("SUBSET tasks with files:")
    for _, r in meta[meta.task_id.isin(ids)].iterrows():
        print("  ", r.task_id, r.file_name, "L", r.Level)
by_ext = collections.defaultdict(list)
for _, r in meta.iterrows():
    by_ext[r.file_name.rsplit(".", 1)[-1]].append(r)
conv = MarkdownConverter()
for ext, rows in sorted(by_ext.items()):
    r = rows[0]
    path = f"{val}/{r.file_name}"
    t0 = time.time()
    try:
        res = conv.convert(path)
        txt = res.text_content
        print(f"[{ext}] OK {r.file_name} {time.time()-t0:.1f}s len={len(txt)} :: {txt[:160]!r}")
    except Exception as e:
        print(f"[{ext}] FAIL {r.file_name}: {type(e).__name__}: {e}")
