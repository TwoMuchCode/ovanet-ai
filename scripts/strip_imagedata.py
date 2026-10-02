import glob, json
for f in glob.glob("data/gold_test_annotations/*.json"):
    d = json.load(open(f, encoding="utf-8"))
    d["imageData"] = None
    json.dump(d, open(f, "w", encoding="utf-8"), indent=2)
print("stripped")