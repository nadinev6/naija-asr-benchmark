import json
n = json.load(open("notebooks/kaggle_benchmark_public.ipynb", encoding="utf-8"))
print(f"cells={len(n['cells'])}, format={n['nbformat']}")
types = {}
for c in n["cells"]:
    t = c["cell_type"]
    types[t] = types.get(t, 0) + 1
    s = "".join(c.get("source", []))[:80].replace("\n", " | ")
    print(f"  [{t}] {s}")
print(f"\nmd={types.get('markdown',0)}, code={types.get('code',0)}")