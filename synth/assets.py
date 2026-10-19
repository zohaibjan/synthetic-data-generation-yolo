"""Reference ingestion and deterministic placement. No uploaded names become paths."""
import hashlib
import io
import json
import random
import re
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES=12*1024*1024

def asset_path(root,asset_id):
    if not re.fullmatch(r"[0-9a-f]{64}",asset_id): raise ValueError("Invalid reference ID")
    path=Path(root)/"assets"/f"{asset_id}.png"
    if not path.is_file() or path.is_symlink(): raise ValueError("Reference image not found")
    return path

def ingest(root,data,kind):
    if len(data)>MAX_BYTES: raise ValueError("Image exceeds 12 MB")
    if kind not in {"scene","object"}: raise ValueError("Choose scene or object")
    try:
        with Image.open(io.BytesIO(data)) as raw:
            if raw.width*raw.height>16_000_000 or min(raw.size)<32: raise ValueError("Image must be 32 pixels or larger and at most 16 megapixels")
            image=ImageOps.exif_transpose(raw).convert("RGBA" if kind=="object" else "RGB")
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as exc:
        raise ValueError("Upload a valid raster image") from exc
    if kind=="object":
        alpha=image.getchannel("A")
        if alpha.getextrema()[0]==255:
            try:
                from rembg import remove
            except ImportError as exc:
                raise ValueError("Upload a transparent PNG cutout, or install the cutout extra for automatic background removal") from exc
            image=remove(image).convert("RGBA")
        bounds=image.getchannel("A").point(lambda a:255 if a>32 else 0).getbbox()
        if not bounds: raise ValueError("Object has no visible pixels")
        if image.getchannel("A").getextrema()[0]==255: raise ValueError("Background removal failed; provide a transparent cutout")
        image=image.crop(bounds)
    image.thumbnail((2048,2048))
    out=io.BytesIO();image.save(out,format="PNG")
    encoded=out.getvalue()
    aid=hashlib.sha256(kind.encode()+encoded).hexdigest()
    folder=Path(root)/"assets";folder.mkdir(exist_ok=True)
    (folder/f"{aid}.png").write_bytes(encoded)
    (folder/f"{aid}.json").write_text(json.dumps({"id":aid,"kind":kind,"width":image.width,"height":image.height,"sha256":hashlib.sha256(encoded).hexdigest()}))
    return json.loads((folder/f"{aid}.json").read_text())

def validate_assets(root,request):
    for kind,ids in (("scene",request.scene_ids),("object",request.object_ids)):
        for aid in ids:
            path=asset_path(root,aid)
            meta=json.loads(path.with_suffix(".json").read_text())
            if meta["kind"]!=kind: raise ValueError("Reference type mismatch")
            if hashlib.sha256(path.read_bytes()).hexdigest()!=meta["sha256"]: raise ValueError("Reference changed after upload")

def prepare_scene(root,request,seed):
    validate_assets(root,request)
    if not request.scene_ids: return None,[],None
    # Round robin guarantees each uploaded scene appears when count >= number of scenes.
    scene=request.scene_ids[(seed-request.seed)%len(request.scene_ids)]
    with Image.open(asset_path(root,scene)) as src:
        image=ImageOps.fit(src.convert("RGB"),(512,512)).convert("RGBA")
    if not request.object_ids: return image.convert("RGB"),[],scene
    rng=random.Random(seed)
    oid=rng.choice(request.object_ids)
    with Image.open(asset_path(root,oid)) as src:
        obj=src.convert("RGBA")
    size=max(12,int(512*request.object_scale*rng.uniform(0.75,1.25)))
    obj.thumbnail((size,size),Image.Resampling.LANCZOS)
    obj=obj.rotate(rng.uniform(0,360),resample=Image.Resampling.BICUBIC,expand=True)
    bounds=obj.getchannel("A").point(lambda a:255 if a>32 else 0).getbbox()
    if not bounds: raise ValueError("Object vanished during resize")
    obj=obj.crop(bounds)
    x=rng.randint(0,512-obj.width);y=rng.randint(0,512-obj.height)
    image.alpha_composite(obj,(x,y))
    box={"class_id":request.object_class,"x":(x+obj.width/2)/512,"y":(y+obj.height/2)/512,"w":obj.width/512,"h":obj.height/512}
    return image.convert("RGB"),[box],scene
