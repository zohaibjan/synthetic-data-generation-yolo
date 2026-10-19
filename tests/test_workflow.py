import io
import json
import zipfile
import pytest
from PIL import Image,ImageDraw
from fastapi.testclient import TestClient
from pydantic import ValidationError
from synth.api import create_app
from synth.models import Request,Box,Review
from synth.assets import ingest,prepare_scene,asset_path
from synth.engine import DemoEngine
from synth.worker import run_job
from synth.export import export_dataset
from synth.store import Store

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('SYNTH_STATE_DIR',str(tmp_path));monkeypatch.setenv('SYNTH_API_TOKEN','test-only')
    return TestClient(create_app(),headers={'Authorization':'Bearer test-only'})

def png(color='blue',cutout=False):
    image=Image.new('RGBA' if cutout else 'RGB',(128,128),(0,0,0,0) if cutout else color)
    if cutout: ImageDraw.Draw(image).rectangle((30,35,90,80),fill='red')
    out=io.BytesIO();image.save(out,format='PNG');return out.getvalue()

def request(**kwargs): return Request(prompt='Aerial car image',classes=['car'],count=4,**kwargs)

def complete(store,req):
    jid=store.submit(req);job=store.claim();assert job['id']==jid
    run_job(store,job,DemoEngine());assert store.job(jid)['status']=='completed'
    return jid

def test_auth(client,monkeypatch):
    assert client.get('/api/jobs',headers={'Authorization':''}).status_code==401
    monkeypatch.delenv('SYNTH_API_TOKEN');assert client.get('/api/jobs').status_code==503

@pytest.mark.parametrize('patch',[{'count':0},{'count':101},{'classes':['car','car']},{'classes':['']},{'strength':0},{'object_ids':['abc']},{'object_class':8},{'guidance':float('nan')}])
def test_request_validation(patch):
    with pytest.raises(ValidationError): Request.model_validate({'prompt':'aerial scene','classes':['car']}|patch)

@pytest.mark.parametrize('patch',[{'x':0},{'w':0},{'class_id':-1},{'x':float('nan')},{'class_id':1.2}])
def test_box_validation(patch):
    with pytest.raises(ValidationError): Box.model_validate({'class_id':0,'x':.5,'y':.5,'w':.2,'h':.2}|patch)

def test_negative_explicit():
    with pytest.raises(ValidationError): Review(status='accepted',boxes=[])
    assert Review(status='accepted',boxes=[],negative_confirmed=True)

def test_upload_limits_and_type(client):
    assert client.post('/api/assets',files={'file':('bad.png',b'no')},data={'kind':'scene'}).status_code==422
    assert client.post('/api/assets',files={'file':('ok.png',png())},data={'kind':'wrong'}).status_code==422
    response=client.post('/api/assets',files={'file':('../../test.png',png())},data={'kind':'scene'})
    assert response.status_code==200
    assert len(response.json()['id'])==64

def test_missing_asset(client):
    response=client.post('/api/jobs',json=request(scene_ids=['a'*64]).model_dump())
    assert response.status_code==422

def test_placement_reproducible_and_in_bounds(tmp_path):
    scene=ingest(tmp_path,png(),'scene')['id'];obj=ingest(tmp_path,png(cutout=True),'object')['id']
    req=request(scene_ids=[scene],object_ids=[obj])
    a,boxes,s=prepare_scene(tmp_path,req,42);b,other,_=prepare_scene(tmp_path,req,42)
    assert a.tobytes()==b.tobytes() and boxes==other and s==scene
    Box.model_validate(boxes[0])
    assert boxes!=prepare_scene(tmp_path,req,43)[1]

def test_source_tampering(tmp_path):
    scene=ingest(tmp_path,png(),'scene')['id'];asset_path(tmp_path,scene).write_bytes(png('red'))
    with pytest.raises(ValueError,match='changed'):prepare_scene(tmp_path,request(scene_ids=[scene]),42)

def test_fifo_restart(tmp_path):
    store=Store(tmp_path);a=store.submit(request());b=store.submit(request())
    assert store.claim()['id']==a
    store.recover();assert store.job(a)['status']=='interrupted'
    assert store.claim()['id']==b
    assert store.claim() is None

def test_job_failure(tmp_path):
    class Broken(DemoEngine):
        def generate(self,*args): raise RuntimeError('controlled failure')
    store=Store(tmp_path);jid=store.submit(request());run_job(store,store.claim(),Broken())
    assert store.job(jid)['status']=='failed'
    with pytest.raises(ValueError):export_dataset(store,jid)

def test_full_reference_review_export(client):
    store=client.app.state.store
    scenes=[ingest(store.root,png(c),'scene')['id'] for c in ['blue','green']]
    obj=ingest(store.root,png(cutout=True),'object')['id']
    req=request(scene_ids=scenes,object_ids=[obj]);jid=client.post('/api/jobs',json=req.model_dump()).json()['id']
    run_job(store,store.claim(),DemoEngine())
    samples=client.get(f'/api/jobs/{jid}/samples').json()
    assert len(samples)==4 and all(s['status']=='pending' for s in samples)
    assert client.get(f'/api/jobs/{jid}/export').status_code==409
    for sample in samples:
        assert client.get(f'/api/samples/{sample["id"]}/image').status_code==200
        assert client.put(f'/api/samples/{sample["id"]}',json={'status':'accepted','boxes':sample['boxes']}).status_code==200
    response=client.get(f'/api/jobs/{jid}/export');assert response.status_code==200
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        manifest=json.loads(z.read('manifest.json'))
        train={s['source_scene'] for s in manifest['samples'] if s['split']=='train'}
        val={s['source_scene'] for s in manifest['samples'] if s['split']=='val'}
        assert train and val and not train&val
        assert len([n for n in z.namelist() if n.endswith('.png')])==4
        assert 'data.yaml' in z.namelist()

def test_single_source_no_leakage(tmp_path):
    store=Store(tmp_path);scene=ingest(tmp_path,png(),'scene')['id']
    jid=complete(store,request(scene_ids=[scene]))
    for s in store.samples(jid):store.review(s['id'],Review(status='accepted',boxes=[],negative_confirmed=True))
    with pytest.raises(ValueError,match='two different'):export_dataset(store,jid)

def test_rejected_not_exported_and_tamper(tmp_path):
    store=Store(tmp_path);jid=complete(store,request())
    samples=store.samples(jid)
    for s in samples[:2]:store.review(s['id'],Review(status='accepted',boxes=[],negative_confirmed=True))
    store.review(samples[2]['id'],Review(status='rejected',boxes=[]))
    with zipfile.ZipFile(io.BytesIO(export_dataset(store,jid))) as z:
        assert len([n for n in z.namelist() if n.endswith('.png')])==2
    store.image_path(samples[0]).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):export_dataset(store,jid)

def test_class_id_and_running_review(client):
    store=client.app.state.store;jid=complete(store,request());s=store.samples(jid)[0]
    bad={'status':'accepted','boxes':[{'class_id':8,'x':.5,'y':.5,'w':.1,'h':.1}]}
    assert client.put(f'/api/samples/{s["id"]}',json=bad).status_code==422
    store.status(jid,'running');assert client.put(f'/api/samples/{s["id"]}',json={'status':'rejected','boxes':[]}).status_code==409

def test_path_traversal(tmp_path):
    with pytest.raises(ValueError):asset_path(tmp_path,'../../etc/passwd')

def test_static_build(client):
    response=client.get('/');assert response.status_code==200
    assert 'Synthetic Data Generation' in response.text


def test_single_scene_training_only_export(tmp_path):
    store=Store(tmp_path);scene=ingest(tmp_path,png(),'scene')['id']
    jid=complete(store,request(scene_ids=[scene]))
    for s in store.samples(jid):store.review(s['id'],Review(status='accepted',boxes=[],negative_confirmed=True))
    with zipfile.ZipFile(io.BytesIO(export_dataset(store,jid,train_only=True))) as z:
        assert 'ADD_REAL_VALIDATION.txt' in z.namelist()
        assert all('/train/' in n for n in z.namelist() if n.endswith('.png'))
        assert 'val:' not in z.read('data.yaml').decode()
