"""GPU adapters are loaded only by the worker, never by the web API."""
import os
from pathlib import Path
from PIL import Image, ImageDraw
from .models import Box

class DemoEngine:
    model_id = "demo-geometric-fixtures-not-stable-diffusion"
    def generate(self, request, seed, initial=None):
        # Test fixture only: no model or generated photographic content.
        if initial is not None: return initial.convert("RGB")
        image = Image.new("RGB", (512,512), (230,225,213))
        draw = ImageDraw.Draw(image)
        x = 50 + seed % 120
        draw.rectangle((x,170,x+150,320),fill=(130,83,60))
        return image
    def label(self, image, classes):
        return []  # Never invent semantic annotations for demo geometry.

class DiffusionEngine:
    def __init__(self):
        import torch
        from diffusers import StableDiffusionPipeline
        self.torch = torch
        model = Path(os.environ["SYNTH_SD_MODEL"]).resolve()
        if not model.is_dir(): raise ValueError("SYNTH_SD_MODEL must be a local Diffusers-format model directory")
        self.device = os.getenv("SYNTH_DEVICE", "cuda:0")
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("Real mode requires CUDA; use demo mode for CPU workflow testing")
        self.model_id = str(model)
        self.pipe = StableDiffusionPipeline.from_pretrained(str(model), torch_dtype=torch.float16, local_files_only=True, use_safetensors=True).to(self.device)
        if self.pipe.safety_checker is None:
            raise RuntimeError("Use a Stable Diffusion checkpoint with its safety checker")
        from diffusers import StableDiffusionImg2ImgPipeline
        self.img2img = StableDiffusionImg2ImgPipeline(**self.pipe.components)
        self.detector = None
        weights = os.getenv("SYNTH_YOLO_MODEL")
        if weights:
            from ultralytics import YOLO
            path = Path(weights).resolve()
            if not path.is_file(): raise ValueError("SYNTH_YOLO_MODEL must be an existing trusted local weight file")
            self.detector = YOLO(str(path))
    def generate(self, request, seed, initial=None):
        kwargs = dict(prompt=request.prompt, negative_prompt=request.negative_prompt,
            num_inference_steps=request.steps,guidance_scale=request.guidance,
            generator=self.torch.Generator(device=self.device).manual_seed(seed))
        if initial is not None:
            # img2img needs at least one denoising step after strength scaling.
            kwargs["num_inference_steps"] = max(request.steps, int(1/request.strength)+1)
            result = self.img2img(image=initial,strength=request.strength,**kwargs)
        else:
            result = self.pipe(width=512,height=512,**kwargs)
        if result.nsfw_content_detected and any(result.nsfw_content_detected):
            raise RuntimeError("Generation filtered by model safety checker; adjust prompt")
        return result.images[0].convert("RGB")
    def label(self,image,classes):
        if self.detector is None: return []
        result = self.detector.predict(image,conf=0.35,device=self.device,verbose=False)[0]
        boxes=[]
        for b in result.boxes:
            name = result.names[int(b.cls.item())]
            if name not in classes: continue
            x,y,w,h = b.xywhn[0].tolist()
            boxes.append(Box(class_id=classes.index(name),x=x,y=y,w=w,h=h).model_dump())
        return boxes
