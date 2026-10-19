"""Contract tests with fake GPU libraries; these do not validate real inference."""
import sys
from types import SimpleNamespace
from PIL import Image
import pytest
from synth.models import Request
from synth.engine import DiffusionEngine

def fake_engine(tmp_path,monkeypatch,filtered=False):
    calls=[]
    class Generator:
        def __init__(self,device): self.device=device
        def manual_seed(self,seed):self.seed=seed;return self
    class Pipe:
        safety_checker=object();components={}
        @classmethod
        def from_pretrained(cls,*args,**kwargs):calls.append(('load',kwargs));return cls()
        def to(self,device):return self
        def __call__(self,**kwargs):calls.append(('generate',kwargs));return SimpleNamespace(images=[Image.new('RGB',(512,512))],nsfw_content_detected=[filtered])
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(float16='float16',cuda=SimpleNamespace(is_available=lambda:True),Generator=Generator))
    monkeypatch.setitem(sys.modules,'diffusers',SimpleNamespace(StableDiffusionPipeline=Pipe,StableDiffusionImg2ImgPipeline=Pipe))
    monkeypatch.setenv('SYNTH_SD_MODEL',str(tmp_path));monkeypatch.delenv('SYNTH_YOLO_MODEL',raising=False)
    return DiffusionEngine(),calls

def test_local_loading_and_seeded_txt2img(tmp_path,monkeypatch):
    engine,calls=fake_engine(tmp_path,monkeypatch)
    req=Request(prompt='Aerial car',classes=['car'])
    engine.generate(req,73)
    assert calls[0][1]['local_files_only'] is True
    assert calls[-1][1]['generator'].seed==73
    assert calls[-1][1]['width']==512

def test_img2img_preserves_input_and_strength(tmp_path,monkeypatch):
    engine,calls=fake_engine(tmp_path,monkeypatch)
    req=Request(prompt='Aerial car',classes=['car'],steps=1,strength=.05)
    image=Image.new('RGB',(512,512));engine.generate(req,1,image)
    args=calls[-1][1];assert args['image'] is image and args['strength']==.05
    assert args['num_inference_steps']*.05>=1

def test_filtered_generation_fails(tmp_path,monkeypatch):
    engine,_=fake_engine(tmp_path,monkeypatch,True)
    with pytest.raises(RuntimeError,match='filtered'):engine.generate(Request(prompt='Aerial car',classes=['car']),1)
