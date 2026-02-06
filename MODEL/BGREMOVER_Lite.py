#--------------------------
#  BG_Remover Node Header
#--------------------------


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from skimage import io, transform, color
from safetensors.torch import load_file
from . import Loader_Lite
from .Loader_Lite import RescaleT, Rescale, ToTensor, ToTensorLab, SalObjDataset 
# ComfyUI 최신 API
from comfy_api.latest import IO, UI

NODE_DIR = os.path.dirname(__file__)

# 모델 상대경로 지정 (커스텀노드 폴더 안)
U2NET_MODEL_PATH = os.path.join(NODE_DIR, "u2net.safetensors")

# 1. safetensors 로드
state_dict = load_file(U2NET_MODEL_PATH)

# 2. 모델 클래스 정의 및 로드
from .u2net import U2NET   # 모델 클래스 정의가 있는 파일
model = U2NET()
model.load_state_dict(state_dict)
model.eval()

# --------------------------
#  Common Preprocessing
# --------------------------

def ensure_mask_tensor(t: torch.Tensor, to_rgb: bool=False) -> torch.Tensor:
    """
    항상 (B,1,H,W) 형태로 보정.
    to_rgb=True일 경우 (B,3,H,W)로 확장.
    """
    if not isinstance(t, torch.Tensor):
        t = torch.from_numpy(np.array(t)).float()

    if t.dim() == 2:          # (H,W)
        t = t.unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
    elif t.dim() == 3:        # (B,H,W)
        t = t.unsqueeze(1)               # (B,1,H,W)
    elif t.dim() == 4:        # (B,C,H,W)
        pass
    else:
        raise ValueError(f"Unsupported mask shape: {t.shape}")

    t = t.float()

    if to_rgb:
        if t.shape[1] == 1:   # (B,1,H,W)
            t = t.repeat(1,3,1,1)   # (B,3,H,W)

    return t



def make_stage(stage_num, num_blocks, num_decoder_blocks=0):
    keys = []
    # 입력 블록
    keys += [f"stage{stage_num}.rebnconvin.{p}" for p in param_list]
    # 인코더 블록
    for i in range(1, num_blocks+1):
        keys += [f"stage{stage_num}.rebnconv{i}.{p}" for p in param_list]
    # 디코더 블록
    for i in range(num_decoder_blocks, 0, -1):
        keys += [f"stage{stage_num}.rebnconv{i}d.{p}" for p in param_list]
    return keys

param_list = [
    "conv_s1.weight","conv_s1.bias",
    "bn_s1.weight","bn_s1.bias",
    "bn_s1.running_mean","bn_s1.running_var"
]

def make_side_outputs(num_sides=6):
    return [f"side{i}.weight" for i in range(1,num_sides+1)] + \
           [f"side{i}.bias" for i in range(1,num_sides+1)] + \
           ["outconv.weight","outconv.bias"]

all_keys = []
all_keys += make_stage(1, 7, 6)   # stage1
all_keys += make_stage(2, 6, 5)   # stage2
all_keys += make_stage(3, 5, 4)   # stage3
all_keys += make_stage(4, 4, 3)   # stage4
all_keys += make_stage(5, 4, 3)   # stage5
all_keys += make_stage(6, 4, 3)   # stage6
all_keys += make_stage("1d", 7, 6) # stage1d (디코더)
all_keys += make_stage("2d", 6, 5) # stage2d
all_keys += make_stage("3d", 5, 4) # stage3d
all_keys += make_stage("4d", 4, 3) # stage4d
all_keys += make_stage("5d", 4, 3) # stage5d
all_keys += make_side_outputs()		   

def preprocess_image(image, size=320, use_lab=True):
    """공통 전처리 함수: 이미지 → 텐서 변환 (RGB+Lab 지원)"""
    # Torch 텐서 처리
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy().astype(np.float32)
    elif isinstance(image, Image.Image):
        img = image.convert("RGB").resize((size, size))
        arr = np.array(img).astype(np.float32) / 255.0
    elif isinstance(image, np.ndarray):
        arr = image.astype(np.float32) / 255.0
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
        
    arr1 = arr
    # Torch 텐서 처리 후 arr 얻은 뒤
    if arr1.ndim == 4:  # (B,H,W,C)
        arr1 = arr1[0]   # 첫 배치만 꺼내서 (H,W,C)
    else:
        pass
    
    arr = arr1


    # 흑백 이미지 처리
    if arr.ndim == 2:  # (H, W)
        arr = np.expand_dims(arr, axis=-1)  # (H, W, 1)
    if arr.shape[-1] == 1:  # 채널이 1개면 3채널로 복제
        arr = np.repeat(arr, 3, axis=-1)

    # 크기 맞추기
    arr = cv2.resize(arr, (size, size))

    # ImageNet 기준 RGB 정규화
    mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    std  = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
    arr = (arr - mean) / std

    # HWC → CHW
    arr = arr.transpose((2, 0, 1))

    tensor = torch.from_numpy(arr).unsqueeze(0).to(dtype=torch.float32)
    return tensor

# --------------------------
#  Auto Nodes
# --------------------------


class BGRemover_Background:
    classname = "BGRemover_Background"
    node_id = "BGRemover_Background"
    DISPLAY_NAME = "배경 중심 자동마스크"
    DESCRIPTION = "U²-Netp 기반 오토마스크 생성. Stage 1~6 중 선택 가능.반전 가능"
    CATEGORY = "리무버/자동"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "입력 이미지"}),
                "stage": (["S0", "S1", "S2", "S3", "S4", "S5", "S6"], {"default": "S1", "tooltip": "Stage 0~6 중 선택 가능"}),
                "amp_factor": ("FLOAT", {"default": 1.000, "min": 1.000, "max": 2.000, "step": 0.001, "tooltip": "출력 증폭 계수"}),
                "invert": (["off", "on"], {"default": "off", "tooltip": "처리반전"}),
                "gamma": ("FLOAT", {"default": 0.600, "min": 0.001, "max": 2.000, "step": 0.001,
                                   "tooltip": "감마 보정 (pow 값)"})
            },
            "optional": {
                "clahe": (["off", "on"], {"default": "off", "tooltip": "클라헤 보정"})
            }
        }


    def __init__(self):
        self.model = model  # 이미 로드된 U2NET 모델 참조

    def execute(self, image, stage="S1", amp_factor=1.000, invert="off", gamma=0.600, clahe="off"):
            # 첫 진입에서 타입 확인 및 변환
        if isinstance(image, Image.Image):
            image = pil_to_tensor(image)  # PIL → Tensor
        elif isinstance(image, np.ndarray):
            image = numpy_to_tensor(image)  # Numpy → Tensor
            # Torch Tensor는 그대로 사용
            # 이후는 항상 텐서로 처리
        t = preprocess_image(image, size=320)

        with torch.no_grad():
            d0, d1, d2, d3, d4, d5, d6 = self.model(t)

        outputs = {"S0": d0, "S1": d1, "S2": d2, "S3": d3, "S4": d4, "S5": d5, "S6": d6}
        mask = torch.sigmoid(outputs.get(stage, d0))
        # 증폭 (amp_factor배)
        mask = torch.clamp(mask * amp_factor, 0, 1)


        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        mask_img = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)

        # CLAHE 적용
        if clahe == "on":
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(32,32))
            mask_clahe = clahe.apply(mask_img)
            mask_tensor = torch.from_numpy(mask_clahe).unsqueeze(0).unsqueeze(0).float() / 255.0

        else:
            # CLAHE를 끄면 원래 mask 그대로 사용
            mask_tensor = torch.from_numpy(mask_img).unsqueeze(0).unsqueeze(0).float() / 255.0
               
        mask = mask_tensor

        # 채널 차원 보장
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

        # 원본 크기 읽기
        if isinstance(image, Image.Image):
            w, h = image.size
        elif isinstance(image, torch.Tensor):
            # ComfyUI IMAGE는 (B,H,W,C)
            h, w = image.shape[1], image.shape[2]
        elif isinstance(image, np.ndarray):
            h, w = image.shape[:2]
            

        

        # 보간 후 soft mask 유지
        soft_mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
        soft_mask = soft_mask.float()

        # 감마 보정
        soft_mask = soft_mask.pow(gamma)
        
        # 반전 옵션
        if invert == "on":
            soft_mask = 1 - soft_mask

        mask = soft_mask.squeeze()# (H,W)

        return (mask,)

# --------------------------

class BGRemoverMaskBackground:
    classname = "BGRemoverMaskBackground"
    node_id = "BGRemoverMaskBackground"
    DISPLAY_NAME = "마스크 중심 자동마스크"
    DESCRIPTION = "U²-Net 기반으로 입력 마스크를 보정/자동화"
    CATEGORY = "리무버/자동"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "입력 마스크"}),
                "stage": (["S0","S1","S2","S3","S4","S5","S6"], {"default":"S1", "tooltip": "Stage 0~6 중 선택 가능"}),
                "amp_factor": ("FLOAT", {"default": 1.000, "min": 1.000, "max": 2.000, "step": 0.001, "tooltip": "출력 증폭 계수"}),
                "invert": (["off", "on"], {"default": "off", "tooltip": "처리반전"}),
                "gamma": ("FLOAT", {"default":0.600,"min":0.001,"max":2.0,"step":0.001, "tooltip": "감마 보정 (pow 값)"}),
            },
            "optional": {
                "clahe": (["off","on"], {"default":"off", "tooltip": "클라헤 보정"})
            }
        }

    def __init__(self):
        self.model = model  # 이미 로드된 U²-Net 모델 참조

    def execute(self, mask, stage="S1", amp_factor=1.000, invert="off", gamma=0.600, clahe="off"):
        # 입력 마스크를 항상 (B,1,H,W) → (B,3,H,W)로 변환
        if mask.dim() == 2:              # (H,W)
            mask_tensor = mask.unsqueeze(0).unsqueeze(0).float()   # (1,1,H,W)
        elif mask.dim() == 3:            # (B,H,W)
            mask_tensor = mask.unsqueeze(1).float()                # (B,1,H,W)
        elif mask.dim() == 4:            # (B,C,H,W)
            mask_tensor = mask.float()
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")

        # 1채널이면 3채널로 확장
        if mask_tensor.shape[1] == 1:
            mask_tensor = mask_tensor.repeat(1,3,1,1)              # (B,3,H,W)
            
        # 원본 크기 저장
        h, w = mask.shape[-2], mask.shape[-1]

        # U²-Net에 넣기 위해 크기 맞추기
        t = F.interpolate(mask_tensor, size=(320,320), mode="bilinear", align_corners=False)

        with torch.no_grad():
            d0,d1,d2,d3,d4,d5,d6 = self.model(t)

        outputs = {"S0":d0,"S1":d1,"S2":d2,"S3":d3,"S4":d4,"S5":d5,"S6":d6}
        mask_out = torch.sigmoid(outputs.get(stage,d0))
        
        mask_out = F.interpolate(mask_out, size=(h,w), mode="bilinear", align_corners=False)

        # 증폭
        mask_out = torch.clamp(mask_out * amp_factor, 0, 1)

        # 정규화
        mask_out = (mask_out - mask_out.min()) / (mask_out.max() - mask_out.min() + 1e-8)

        # CLAHE 적용
        mask_img = (mask_out.squeeze().cpu().numpy() * 255).astype(np.uint8)
        if clahe == "on":
            clahe_obj = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(32,32))
            mask_img = clahe_obj.apply(mask_img)
        mask_tensor = torch.from_numpy(mask_img).unsqueeze(0).unsqueeze(0).float() / 255.0

        # 감마 보정
        mask_tensor = mask_tensor.pow(gamma)

        # 반전
        if invert == "on":
            mask_tensor = 1 - mask_tensor

        # 최종 차원 보정
        if mask_tensor.dim() == 4:        # (1,1,H,W)
            mask_tensor = mask_tensor.squeeze(0).squeeze(0)   # (H,W)
        elif mask_tensor.dim() == 3:      # (1,H,W)
            mask_tensor = mask_tensor.squeeze(0)              # (H,W)

        return (mask_tensor,)

class BGRemoverMaskBackground:
    classname = "BGRemoverMaskBackground"
    node_id = "BGRemoverMaskBackground"
    DISPLAY_NAME = "마스크 중심 자동마스크"
    DESCRIPTION = "U²-Net 기반으로 입력 마스크를 보정/자동화"
    CATEGORY = "리무버/자동"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "입력 마스크"}),
                "stage": (["S0","S1","S2","S3","S4","S5","S6"], {"default":"S1", "tooltip": "Stage 0~6 중 선택 가능"}),
                "amp_factor": ("FLOAT", {"default": 1.000, "min": 1.000, "max": 2.000, "step": 0.001, "tooltip": "출력 증폭 계수"}),
                "invert": (["off", "on"], {"default": "off", "tooltip": "처리반전"}),
                "gamma": ("FLOAT", {"default":0.600,"min":0.001,"max":2.0,"step":0.001, "tooltip": "감마 보정 (pow 값)"}),
            },
            "optional": {
                "clahe": (["off","on"], {"default":"off", "tooltip": "클라헤 보정"})
            }
        }

    def __init__(self):
        self.model = model  # 이미 로드된 U²-Net 모델 참조

    def execute(self, mask, stage="S1", amp_factor=1.000, invert="off", gamma=0.600, clahe="off"):
        # 입력 마스크를 항상 (B,1,H,W) → (B,3,H,W)로 변환
        if mask.dim() == 2:              # (H,W)
            mask_tensor = mask.unsqueeze(0).unsqueeze(0).float()   # (1,1,H,W)
        elif mask.dim() == 3:            # (B,H,W)
            mask_tensor = mask.unsqueeze(1).float()                # (B,1,H,W)
        elif mask.dim() == 4:            # (B,C,H,W)
            mask_tensor = mask.float()
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")

        # 1채널이면 3채널로 확장
        if mask_tensor.shape[1] == 1:
            mask_tensor = mask_tensor.repeat(1,3,1,1)              # (B,3,H,W)
            
        # 원본 크기 저장
        h, w = mask.shape[-2], mask.shape[-1]

        # U²-Net에 넣기 위해 크기 맞추기
        t = F.interpolate(mask_tensor, size=(320,320), mode="bilinear", align_corners=False)

        with torch.no_grad():
            d0,d1,d2,d3,d4,d5,d6 = self.model(t)

        outputs = {"S0":d0,"S1":d1,"S2":d2,"S3":d3,"S4":d4,"S5":d5,"S6":d6}
        mask_out = torch.sigmoid(outputs.get(stage,d0))
        
        mask_out = F.interpolate(mask_out, size=(h,w), mode="bilinear", align_corners=False)

        # 증폭
        mask_out = torch.clamp(mask_out * amp_factor, 0, 1)

        # 정규화
        mask_out = (mask_out - mask_out.min()) / (mask_out.max() - mask_out.min() + 1e-8)

        # CLAHE 적용
        mask_img = (mask_out.squeeze().cpu().numpy() * 255).astype(np.uint8)
        if clahe == "on":
            clahe_obj = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(32,32))
            mask_img = clahe_obj.apply(mask_img)
        mask_tensor = torch.from_numpy(mask_img).unsqueeze(0).unsqueeze(0).float() / 255.0

        # 감마 보정
        mask_tensor = mask_tensor.pow(gamma)

        # 반전
        if invert == "on":
            mask_tensor = 1 - mask_tensor

        # 최종 차원 보정
        if mask_tensor.dim() == 4:        # (1,1,H,W)
            mask_tensor = mask_tensor.squeeze(0).squeeze(0)   # (H,W)
        elif mask_tensor.dim() == 3:      # (1,H,W)
            mask_tensor = mask_tensor.squeeze(0)              # (H,W)

        return (mask_tensor,)

# --------------------------
#  Semi-Auto Nodes
# --------------------------

class BGRemover_BackgroundAdv:
    classname = "BGRemover_BackgroundAdv"
    node_id = "BGRemover_BackgroundAdv"
    DISPLAY_NAME = "배경 중심 반자동 마스크"
    DESCRIPTION = "U²-Net 기반 오토마스크 생성 후 유저마스크로 보정/합성"
    CATEGORY = "리무버/반자동"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "입력 이미지"}),
                "stage": (["S0", "S1", "S2", "S3", "S4", "S5", "S6"], {"default": "S1", "tooltip": "Stage 0~6 중 선택 가능"}),
                "amp_factor": ("FLOAT", {"default": 1.000, "min": 1.000, "max": 2.000, "step": 0.001, "tooltip": "출력 증폭 계수"}),
                "invert": (["off", "on"], {"default": "off", "tooltip": "처리반전"}),
                "gamma": ("FLOAT", {"default": 0.600, "min": 0.001, "max": 2.000, "step": 0.001, "tooltip": "감마 보정 (pow 값)"}),
            },
            "optional": {
                "clahe": (["off", "on"], {"default": "off", "tooltip": "클라헤 보정"}),
                "user_mask": ("MASK", {"default": None, "tooltip": "사용자 추가 마스크"}),
                "blend_mode": (["max", "average", "overwrite"], {"default": "max", "tooltip": "마스크 합성 방식"})
            }

        }

    def __init__(self):
        self.model = model  # 이미 로드된 U2NET 모델 참조

    def execute(self, image, stage="S1", amp_factor=1.000, invert="off", gamma=0.600, clahe="off", user_mask=None, blend_mode="max"):
        if self.model is None:
            if isinstance(image, torch.Tensor):
                h, w = image.shape[1], image.shape[2]
            elif isinstance(image, np.ndarray):
                h, w = image.shape[:2]
            else:  # PIL
                w, h = image.size
            return (torch.zeros(1, 1, h, w),)

        t = preprocess_image(image, size=320)

        with torch.no_grad():
            d0, d1, d2, d3, d4, d5, d6 = self.model(t)

        outputs = {"S0": d0, "S1": d1, "S2": d2, "S3": d3, "S4": d4, "S5": d5, "S6": d6}
        mask = torch.sigmoid(outputs.get(stage, d0))
        # 증폭 (amp_factor배)
        mask = torch.clamp(mask * amp_factor, 0, 1)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        mask_img = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)

        # CLAHE 적용
        if clahe == "on":
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(32,32))
            mask_clahe = clahe.apply(mask_img)
            mask_tensor = torch.from_numpy(mask_clahe).unsqueeze(0).unsqueeze(0).float() / 255.0

        else:
            # CLAHE를 끄면 원래 mask 그대로 사용
            mask_tensor = torch.from_numpy(mask_img).unsqueeze(0).unsqueeze(0).float() / 255.0
               
        mask = mask_tensor

        # 채널 차원 보장
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

        # 원본 크기 읽기
        if isinstance(image, Image.Image):
            w, h = image.size
        elif isinstance(image, torch.Tensor):
            # ComfyUI IMAGE는 (B,H,W,C)
            h, w = image.shape[1], image.shape[2]
        elif isinstance(image, np.ndarray):
            h, w = image.shape[:2]
            

        

        # 보간 후 soft mask 유지
        soft_mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
        mask = soft_mask.float()
        
        # 사용자 마스크와 합성 (합집합)
        if user_mask is not None:    
            # 크기 맞추기
            if user_mask.shape[-2:] != mask.shape[-2:]:
                user_mask = F.interpolate(
                    user_mask.unsqueeze(0) if user_mask.ndim == 2 else user_mask,
                    size=mask.shape[-2:], mode="bilinear", align_corners=False
                )

            if user_mask.ndim == 2:  # (H,W)
                user_mask = user_mask.unsqueeze(0).unsqueeze(0)
            elif user_mask.ndim == 3 and user_mask.shape[0] == 1:  # (1,H,W)
                user_mask = user_mask.unsqueeze(1)
            user_mask = user_mask.float()
            
            if blend_mode == "max":
                mask = torch.max(mask, user_mask)
            elif blend_mode == "average":
                mask = (mask + user_mask) / 2
            elif blend_mode == "overwrite":
                mask = torch.where(user_mask > 0.5, torch.ones_like(mask), mask)

        else:
            # 연결 안 된 경우는 그냥 U²Net 마스크만 사용
            mask = mask


        # 감마 보정
        if gamma != 1.0:
            mask = mask.pow(gamma)

        # 반전 옵션
        if invert == "on":
            mask = 1 - mask

        # 원본 크기로 보간
        if isinstance(image, torch.Tensor):
            h, w = image.shape[1], image.shape[2]
        elif isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:  # PIL
            w, h = image.size

        mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
        return (mask.squeeze(),)

# --------------------------


class BGRemoverMaskBackgroundAdv:
    classname = "BGRemoverMaskBackgroundAdv"
    node_id = "BGRemoverMaskBackgroundAdv"
    DISPLAY_NAME = "배경 중심 반자동마스크(마스크)"
    DESCRIPTION = "U²-Net 기반 오토마스크 생성 후 입력마스크와 합성/보정"
    CATEGORY = "리무버/반자동"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "입력 마스크"}),
                "stage": (["S0","S1","S2","S3","S4","S5","S6"], {"default":"S1", "tooltip": "Stage 0~6 중 선택 가능"}),
                "amp_factor": ("FLOAT", {"default":1.000,"min":1.000,"max":2.000,"step":0.001, "tooltip": "출력 증폭 계수"}),
                "invert": (["off","on"], {"default":"off", "tooltip": "처리반전"}),
                "gamma": ("FLOAT", {"default":0.600,"min":0.001,"max":2.000,"step":0.001,  "tooltip": "감마 보정 (pow 값)"}),
            },
            "optional": {
                "clahe": (["off", "on"], {"default": "off", "tooltip": "클라헤 보정"}),
                "user_mask": ("MASK", {"default": None, "tooltip": "사용자 추가 마스크"}),
                "blend_mode": (["max", "average", "overwrite"], {"default": "max", "tooltip": "마스크 합성 방식"})
            }
        }

    def __init__(self):
        self.model = model  # U²Net 모델 참조

    def execute(self, mask, stage="S1", amp_factor=1.000, invert="off", gamma=0.600,
                clahe="off", user_mask=None, blend_mode="max"):

        # 1. 마스크 → RGB 변환
        if mask.dim() == 2:              # (H,W)
            mask_tensor = mask.unsqueeze(0).unsqueeze(0).float()
        elif mask.dim() == 3:            # (B,H,W)
            mask_tensor = mask.unsqueeze(1).float()
        elif mask.dim() == 4:            # (B,C,H,W)
            mask_tensor = mask.float()
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")

        if mask_tensor.shape[1] == 1:
            mask_tensor = mask_tensor.repeat(1,3,1,1)  # (B,3,H,W)

        h, w = mask.shape[-2], mask.shape[-1]

        # 2. 리사이즈 (320×320)
        t = F.interpolate(mask_tensor, size=(320,320), mode="bilinear", align_corners=False)

        # 3. 모델 추론
        with torch.no_grad():
            d0,d1,d2,d3,d4,d5,d6 = self.model(t)
        outputs = {"S0":d0,"S1":d1,"S2":d2,"S3":d3,"S4":d4,"S5":d5,"S6":d6}
        pred = torch.sigmoid(outputs.get(stage,d0))

        # 4. 원본 크기 복원
        mask_out = F.interpolate(pred, size=(h,w), mode="bilinear", align_corners=False)

        # 5. 증폭/정규화
        mask_out = torch.clamp(mask_out * amp_factor, 0, 1)
        mask_out = (mask_out - mask_out.min()) / (mask_out.max() - mask_out.min() + 1e-8)

        # 6. CLAHE
        mask_img = (mask_out.squeeze().cpu().numpy() * 255).astype(np.uint8)
        if clahe == "on":
            clahe_obj = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(32,32))
            mask_img = clahe_obj.apply(mask_img)
        mask_tensor = torch.from_numpy(mask_img).unsqueeze(0).unsqueeze(0).float() / 255.0

        # 7. 유저마스크 합성
        if user_mask is not None:
            if user_mask.ndim == 2:
                user_mask = user_mask.unsqueeze(0).unsqueeze(0)
            elif user_mask.ndim == 3 and user_mask.shape[0] == 1:
                user_mask = user_mask.unsqueeze(1)
            user_mask = user_mask.float()

            if blend_mode == "max":
                mask_tensor = torch.max(mask_tensor, user_mask)
            elif blend_mode == "average":
                mask_tensor = (mask_tensor + user_mask) / 2
            elif blend_mode == "overwrite":
                mask_tensor = torch.where(user_mask > 0.5, torch.ones_like(mask_tensor), mask_tensor)

        # 8. 감마 보정
        mask_tensor = mask_tensor.pow(gamma)

        # 9. 반전
        if invert == "on":
            mask_tensor = 1 - mask_tensor

        return (mask_tensor.squeeze(),)




#--------------------------
#  CLASS_MAPPINGS
#--------------------------

NODE_CLASS_MAPPINGS = {
    "BGRemover_Background": BGRemover_Background,
    "BGRemoverMaskBackground": BGRemoverMaskBackground,
    "BGRemover_BackgroundAdv": BGRemover_BackgroundAdv,
    "BGRemoverMaskBackgroundAdv": BGRemoverMaskBackgroundAdv,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "BGRemover_Background": "배경 중심 자동마스크",
    "BGRemoverMaskBackground": "배경 중심 자동마스크(마스크)",
    "BGRemover_BackgroundAdv": "배경 중심 반자동 마스크",
    "BGRemoverMaskBackgroundAdv": "배경 중심 반자동마스크(마스크)",
}
