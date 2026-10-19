from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator

class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=5, max_length=1500)
    negative_prompt: str = Field(default="blurry, watermark, text, distorted", max_length=1500)
    classes: list[str] = Field(min_length=1, max_length=80)
    scene_ids: list[str] = Field(default_factory=list, max_length=50)
    object_ids: list[str] = Field(default_factory=list, max_length=50)
    object_class: int = Field(default=0, ge=0)
    strength: float = Field(default=0.3, ge=0.05, le=0.75, allow_inf_nan=False)
    object_scale: float = Field(default=0.15, ge=0.03, le=0.5, allow_inf_nan=False)
    @model_validator(mode="after")
    def references(self):
        if self.object_ids and not self.scene_ids:
            raise ValueError("Object placement requires at least one scene")
        if self.object_class >= len(self.classes):
            raise ValueError("Object class index is out of range")
        return self
    count: int = Field(default=8, ge=2, le=100)
    seed: int = Field(default=42, ge=0, le=2**31-101)
    steps: int = Field(default=25, ge=1, le=100)
    guidance: float = Field(default=7.5, ge=1, le=20, allow_inf_nan=False)
    @field_validator("classes")
    @classmethod
    def classes_valid(cls, names):
        names = [n.strip() for n in names]
        if any(not n or len(n)>80 for n in names) or len(set(names)) != len(names):
            raise ValueError("Class names must be nonempty and unique")
        return names

class Box(BaseModel):
    model_config = ConfigDict(extra="forbid")
    class_id: int = Field(ge=0, strict=True)
    x: float = Field(ge=0, le=1, allow_inf_nan=False)
    y: float = Field(ge=0, le=1, allow_inf_nan=False)
    w: float = Field(gt=0, le=1, allow_inf_nan=False)
    h: float = Field(gt=0, le=1, allow_inf_nan=False)
    @model_validator(mode="after")
    def inside(self):
        if self.x-self.w/2 < -1e-7 or self.y-self.h/2 < -1e-7 or self.x+self.w/2 > 1+1e-7 or self.y+self.h/2 > 1+1e-7:
            raise ValueError("Box must fit within image")
        return self

class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["accepted", "rejected", "pending"]
    boxes: list[Box] = Field(max_length=200)
    negative_confirmed: bool = False
    @model_validator(mode="after")
    def empty_is_explicit(self):
        if self.status == "accepted" and not self.boxes and not self.negative_confirmed:
            raise ValueError("Explicitly confirm a negative image with no target objects")
        return self
