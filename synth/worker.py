import fcntl
import hashlib
import json
import os
import time
from .models import Request
from .store import Store
from .assets import prepare_scene
from .engine import DemoEngine, DiffusionEngine

def run_job(store,job,engine):
    request = Request.model_validate(job["request"])
    folder=store.root / "images" / job["id"]
    folder.mkdir(parents=True,exist_ok=True)
    try:
        (folder / "provenance.json").write_text(json.dumps({"model":engine.model_id,"mode":os.getenv("SYNTH_MODE","real"),"request":request.model_dump(),"label_source":"optional teacher predictions, human review required"},indent=2))
        for seed in range(request.seed,request.seed+request.count):
            initial, placed, source_scene = prepare_scene(store.root, request, seed)
            image=engine.generate(request,seed,initial)
            # Placement boxes are proposals after diffusion, not ground truth.
            boxes=placed if placed else engine.label(image,request.classes)
            path=folder / f"{seed}.png"
            image.save(path)
            store.add_sample(job["id"],seed,boxes,hashlib.sha256(path.read_bytes()).hexdigest(),source_scene)
        store.status(job["id"],"completed")
    except Exception as exc:
        store.status(job["id"],"failed",f"{type(exc).__name__}: {exc}"[:1000])

def main():
    store=Store(os.getenv("SYNTH_STATE_DIR","./state"))
    # One worker per state directory, one in-flight diffusion call. Lock lives for process lifetime.
    with open(store.root/"worker.lock","a") as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit("A worker is already running")
        store.recover()
        mode=os.getenv("SYNTH_MODE","real")
        if mode not in {"real","demo"}: raise SystemExit("SYNTH_MODE must be real or demo")
        engine=DemoEngine() if mode=="demo" else DiffusionEngine()
        while True:
            job=store.claim()
            if job: run_job(store,job,engine)
            else: time.sleep(1)
if __name__=="__main__": main()
