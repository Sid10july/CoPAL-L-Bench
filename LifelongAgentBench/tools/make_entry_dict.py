# tools/make_entry_dict.py
import argparse, json, ast
from pathlib import Path
import pandas as pd

def parse(v):
    if isinstance(v,(dict,list)): return v
    if isinstance(v,str):
        s=v.strip()
        for fn in (json.loads, ast.literal_eval):
            try: return fn(s)
            except: pass
    return v

def pick(r,names):
    for n in names:
        if n in r and pd.notna(r[n]): return r[n]
    return None

p=argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--outdir", default="data/v0303/db_bench/processed/v0317_first500")
p.add_argument("--subset", type=int, default=500)
a=p.parse_args()

files=sorted(Path(a.src).rglob("*.parquet"))
if not files: raise SystemExit("no parquet under --src")
df=pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
key="sample_index" if "sample_index" in df.columns else None
if key is None:
    df=df.reset_index().rename(columns={"index":"sample_index"})
    key="sample_index"
df=df.sort_values(by=key, kind="stable")

out={}
for _, r in df.iterrows():
    k=str(int(r[key])) if pd.notna(r[key]) else str(len(out))
    inst=pick(r,["instruction","prompt","query"])
    tbl=parse(pick(r,["table_info","table","table_json"]))
    ans=parse(pick(r,["answer_info","answer","answer_json"]))
    skills=parse(pick(r,["skill_list","skills"]))
    h=pick(r,["sql_instruction_row_list_entry_hash","hash"])
    if not isinstance(ans,dict) or "md5" not in ans: 
        continue
    out[k]={"instruction":inst,"table_info":tbl,"answer_info":ans,"skill_list":skills,"hash":h}
    if len(out)>=a.subset: break

Path(a.outdir).mkdir(parents=True, exist_ok=True)
with open(Path(a.outdir)/"entry_dict.json","w") as f: json.dump(out,f)
print(Path(a.outdir)/"entry_dict.json")
