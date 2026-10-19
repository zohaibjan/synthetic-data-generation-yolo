import hashlib
import io
import json
import random
import zipfile
import yaml
from .models import Review

def export_dataset(store,jid,train_only=False):
    job=store.job(jid)
    if not job: raise ValueError("Job not found")
    if job["status"] != "completed": raise ValueError("Only completed jobs can be exported")
    samples=[s for s in store.samples(jid) if s["status"]=="accepted"]
    if len(samples)<2: raise ValueError("Accept at least two reviewed images before export")
    random.Random(job["request"]["seed"]).shuffle(samples)
    sources=sorted({s["source_scene"] for s in samples if s["source_scene"]})
    if not train_only and sources and len(sources)<2:
        raise ValueError("Accept images from at least two different scenery images for a scene-disjoint train/val split")
    random.Random(job["request"]["seed"]).shuffle(sources)
    val_sources=set(sources[:max(1,round(len(sources)*0.2))])
    val_count=max(1,round(len(samples)*0.2))
    manifest={"job":job,"synthetic":True,"warning":"Synthetic validation is a smoke test only. Evaluate on an independent real-world test set.","samples":[]}
    output=io.BytesIO()
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as z:
        config={"train":"images/train","names":dict(enumerate(job["request"]["classes"]))}
        if not train_only: config["val"]="images/val"
        else:
            z.writestr("ADD_REAL_VALIDATION.txt","Training-only export. Add an independent labelled real validation set, then set val in data.yaml before using Ultralytics or Vision Training Hub. Do not duplicate training images as validation.")
        z.writestr("data.yaml",yaml.safe_dump(config))
        for i,s in enumerate(samples):
            review=Review(status="accepted",boxes=s["boxes"],negative_confirmed=bool(s["negative"]))
            if any(b.class_id>=len(job["request"]["classes"]) for b in review.boxes): raise ValueError("Invalid class index")
            path=store.image_path(s)
            if path.is_symlink(): raise ValueError("Image symlinks are not allowed")
            data=path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=s["sha256"]: raise ValueError("Image changed after generation")
            split=("val" if s["source_scene"] in val_sources else "train") if sources else ("val" if i<val_count else "train")
            if train_only: split="train"
            z.writestr(f'images/{split}/{s["id"]}.png',data)
            labels="\n".join(f'{b.class_id} {b.x:.8f} {b.y:.8f} {b.w:.8f} {b.h:.8f}' for b in review.boxes)
            z.writestr(f'labels/{split}/{s["id"]}.txt',labels+"\n" if labels else "")
            manifest["samples"].append(s | {"split":split})
        provenance=store.root/"images"/jid/"provenance.json"
        manifest["generation"]=json.loads(provenance.read_text())
        z.writestr("manifest.json",json.dumps(manifest,indent=2))
    return output.getvalue()
