import hmac
import os
from pathlib import Path
from fastapi import FastAPI, Depends, Header, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from .assets import ingest, validate_assets, MAX_BYTES
from .store import Store
from .models import Request, Review
from .export import export_dataset

def create_app():
    app=FastAPI(title="Synthetic Data Generation for YOLO")
    store=Store(os.getenv("SYNTH_STATE_DIR","./state"))
    app.state.store=store
    def auth(authorization: str=Header(default="")):
        token=os.getenv("SYNTH_API_TOKEN","")
        if not token: raise HTTPException(503,"Set SYNTH_API_TOKEN on the server")
        if not hmac.compare_digest(authorization, "Bearer "+token): raise HTTPException(401,"Invalid API token")
    def job(jid):
        value=store.job(jid)
        if not value: raise HTTPException(404,"Job not found")
        return value
    def sample(sid):
        value=store.sample(sid)
        if not value: raise HTTPException(404,"Image not found")
        return value
    @app.post("/api/assets",dependencies=[Depends(auth)])
    def upload(file:UploadFile=File(...),kind:str=Form(...)):
        try:
            data=file.file.read(MAX_BYTES+1)
            return ingest(store.root,data,kind)
        except ValueError as exc: raise HTTPException(422,str(exc)) from exc
        finally: file.file.close()
    @app.get("/api/jobs",dependencies=[Depends(auth)])
    def jobs(): return store.jobs()
    @app.post("/api/jobs",status_code=201,dependencies=[Depends(auth)])
    def submit(request: Request):
        try: validate_assets(store.root,request)
        except (ValueError,OSError) as exc: raise HTTPException(422,str(exc)) from exc
        return {"id":store.submit(request)}
    @app.get("/api/jobs/{jid}/samples",dependencies=[Depends(auth)])
    def samples(jid:str):
        job(jid)
        return store.samples(jid)
    @app.get("/api/samples/{sid}/image",dependencies=[Depends(auth)])
    def image(sid:str):
        path=store.image_path(sample(sid))
        if not path.is_file() or path.is_symlink(): raise HTTPException(404,"Image unavailable")
        return FileResponse(path,media_type="image/png",headers={"Cache-Control":"no-store"})
    @app.put("/api/samples/{sid}",dependencies=[Depends(auth)])
    def review(sid:str,review:Review):
        s=sample(sid)
        j=job(s["job"])
        if j["status"]!="completed": raise HTTPException(409,"Wait for generation to complete")
        if any(b.class_id>=len(j["request"]["classes"]) for b in review.boxes): raise HTTPException(422,"Unknown class index")
        store.review(sid,review)
        return store.sample(sid)
    @app.get("/api/jobs/{jid}/export",dependencies=[Depends(auth)])
    def export(jid:str,train_only:bool=False):
        job(jid)
        try: data=export_dataset(store,jid,train_only)
        except (ValueError,OSError) as exc: raise HTTPException(409,str(exc)) from exc
        return Response(data,media_type="application/zip",headers={"Content-Disposition":f'attachment; filename="synthetic-{jid}.zip"'})
    static=Path(__file__).parent/"static"
    if static.exists(): app.mount("/",StaticFiles(directory=static,html=True),name="frontend")
    return app
