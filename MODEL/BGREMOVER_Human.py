# -----------------------------------------
#  BG_Remover Human Node Header
# -----------------------------------------


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from skimage import io, transform, color
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from . import Loader_Lite
from .Loader_Lite import RescaleT, Rescale, ToTensor, ToTensorLab, SalObjDataset 
# ComfyUI 최신 API
from comfy_api.latest import IO, UI

NODE_DIR = os.path.dirname(__file__)

# 모델 상대경로 지정 (커스텀노드 폴더 안)
U2NET_MODEL_PATH = os.path.join(NODE_DIR, "u2net.safetensors")
U2NET_HUMAN_MODEL_PATH = os.path.join(NODE_DIR, "u2net_human_seg.safetensors")

# 1. safetensors 로드
state_dict_u2net = load_file(U2NET_MODEL_PATH)
state_dict_human = load_file(U2NET_HUMAN_MODEL_PATH)

# 2. 모델 클래스 정의 및 로드
from .u2net import U2NET   # 모델 클래스 정의가 있는 파일
from .u2net_Human_Seg import normPRED

# 기본 U²Net
u2net_model = U2NET()
u2net_model.load_state_dict(state_dict_u2net)
u2net_model.eval()

# 휴먼 전용 U²Net

u2net_human_model = U2NET()
u2net_human_model.load_state_dict(state_dict_human)
u2net_human_model.eval()


# -----------------------------------------
#  Common Preprocessing
# -----------------------------------------

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
    """공통 전처리 함수: 이미지 → 텐서 변환 (RGB)"""
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

def dilate_tensor(mask_tensor, kernel_size=5, iterations=1):
    """
    PyTorch 기반 딜레이션 (max pooling 활용)
    mask_tensor: (B,1,H,W) float tensor
    """
    for _ in range(iterations):
        mask_tensor = F.max_pool2d(mask_tensor, kernel_size, stride=1, padding=kernel_size//2)
    return mask_tensor

def blur_tensor(mask_tensor, k=5):
    """
    PyTorch 기반 블러 (avg pooling으로 근사)
    mask_tensor: (B,1,H,W) float tensor
    """
    return F.avg_pool2d(mask_tensor, kernel_size=k, stride=1, padding=k//2)


# -----------------------------------------
#  Auto Nodes
# -----------------------------------------


class BGRemover_Human:
    classname = "BGRemover_Human"
    node_id = "BGRemover_Human"
    DISPLAY_NAME = "인물 중심 자동마스크"
    DESCRIPTION = "U²-Net Human 기반 오토마스크 생성. Stage 1(d2), 2(d3),  3(d4) 선택 가능"
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
                "clahe": (["off", "on"], {"default": "off", "tooltip": "클라헤 보정"}),
                "dilate_iterations": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1, "tooltip": "마스크 팽창 횟수"}),
                "blur_strength": (["0","1","2","3","4","5","6"], {"default": "0", "tooltip": "블러 강도(0=없음)"})
            }
        }

    def __init__(self):
        self.model = u2net_human_model  # 사람 전용 모델 참조

    def execute(self, image, stage="S1", amp_factor=1.000, invert="off", gamma=0.600, clahe="off", dilate_iterations=0, blur_strength="0"):
        t = preprocess_image(image, size=320)

        with torch.no_grad():
            d0, d1, d2, d3, d4, d5, d6 = self.model(t)

        # stage=0 → d0, stage=1 → d1, stage=2 → d2,stage=3 → d3, stage=4 → d4, stage=5 → d5, stage=6 → d6
        outputs = {"S0": d0, "S1": d1, "S2": d2, "S3": d3, "S4": d4, "S5": d5, "S6": d6}
        pred  = torch.sigmoid(outputs.get(stage, d0))
        # 증폭 (amp_factor배)
        pred = torch.clamp(pred * amp_factor, 0, 1)
        mask = normPRED(pred)   # 보조 함수 적용
        
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        
        # CLAHE
        if clahe == "on":
            mask_img = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
            clahe_obj = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(32,32))
            mask_img = clahe_obj.apply(mask_img)
            mask_tensor = torch.from_numpy(mask_img).unsqueeze(0).unsqueeze(0).float() / 255.0

        else:
            # CLAHE가 꺼져 있으면 float 그대로 유지
            mask_tensor = mask   # (1,1,H,W)


        # 딜레이션
        if dilate_iterations > 0:
            mask_tensor = dilate_tensor(mask_tensor, kernel_size=5, iterations=dilate_iterations)


        # 블러
        blur_map = {"0":None,"1":3,"2":5,"3":7,"4":11,"5":13,"6":15}
        if blur_strength != "0":
            k = blur_map[blur_strength]
            mask_tensor = blur_tensor(mask_tensor, k)


        # Torch 변환
        mask = mask_tensor


        # 원본 크기 맞추기
        if isinstance(image, torch.Tensor):
            h, w = image.shape[1], image.shape[2]
        elif isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:  # PIL
            w, h = image.size

        soft_mask = F.interpolate(mask_tensor, size=(h, w),
                                  mode="bilinear", align_corners=False).float()

        # 감마 보정
        soft_mask = soft_mask.pow(gamma)

        # 반전 옵션
        if invert == "on":
            soft_mask = 1 - soft_mask

        return (soft_mask.squeeze(),)




#---------------------------

class BGRemoverMaskHuman:
    classname = "BGRemoverMaskHuman"
    node_id = "BGRemoverMaskHuman"
    DISPLAY_NAME = "인물 중심 제거(마스크)"
    DESCRIPTION = "U²-Net Human 기반 오토마스크 영역 생성."
    CATEGORY = "리무버/자동"
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
                "gamma": ("FLOAT", {"default":0.600,"min":0.001,"max":2.000,"step":0.001, "tooltip": "감마 보정 (pow 값)"})
            },
            "optional": {
                "clahe": (["off","on"], {"default":"off", "tooltip": "클라헤 보정"}),
                "dilate_iterations": ("INT", {"default":0,"min":0,"max":10,"step":1, "tooltip": "마스크 팽창 횟수"}),
                "blur_strength": (["0","1","2","3","4","5","6"], {"default":"0", "tooltip": "블러 강도(0=없음)"})
            }
        }
        
    def __init__(self):
        self.model = u2net_human_model  # 휴먼 전용 모델 참조

    def execute(self, mask, stage="S1", amp_factor=1.000, invert="off", gamma=0.600,
                clahe="off", dilate_iterations=0, blur_strength="0"):

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

        # 7. 팽창
        if dilate_iterations > 0:
            mask_tensor = dilate_tensor(mask_tensor, kernel_size=5, iterations=dilate_iterations)

        # 8. 블러
        blur_map = {"0":None,"1":3,"2":5,"3":7,"4":11,"5":13,"6":15}
        if blur_strength != "0":
            k = blur_map[blur_strength]
            mask_tensor = blur_tensor(mask_tensor, k)

        # 9. 감마 보정
        mask_tensor = mask_tensor.pow(gamma)

        # 10. 반전
        if invert == "on":
            mask_tensor = 1 - mask_tensor

        return (mask_tensor.squeeze(),)




# --------------------------
#  Semi-Auto Nodes
# --------------------------

class BGRemover_HumanAdv:
    classname = "BGRemover_HumanAdv"
    node_id = "BGRemover_HumanAdv"
    DISPLAY_NAME = "인물 중심 반자동 마스크"
    DESCRIPTION = "U²-Net Human 기반 마스크 생성 후 유저마스크로 보정/합성. Stage 1(d2), 2(d3),  3(d4) 선택 가능"
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
                "blend_mode": (["max", "average", "overwrite"], {"default": "max", "tooltip": "마스크 합성 방식"}),
                "dilate_iterations": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1, "tooltip": "마스크 팽창 횟수"}),
                "blur_strength": (["0","1","2","3","4","5","6"], {"default": "0", "tooltip": "블러 강도(0=없음)"})
            }
        }
    def __init__(self):
        self.model = u2net_human_model

    def execute(self, image, stage="S1", amp_factor=1.000, invert="off", gamma=0.600,
                clahe="off", user_mask=None, blend_mode="max",
                dilate_iterations=0, blur_strength="0"):

        t = preprocess_image(image, size=320)
        with torch.no_grad():
            d0,d1,d2,d3,d4,d5,d6 = self.model(t)

        outputs = {"S0":d0,"S1":d1,"S2":d2,"S3":d3,"S4":d4,"S5":d5,"S6":d6}
        pred = torch.sigmoid(outputs.get(stage,d0))
        # 증폭 (amp_factor배)
        pred = torch.clamp(pred * amp_factor, 0, 1)
        mask = normPRED(pred)

        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        print("mask shape MinMax:", mask.shape)
        
        # CLAHE
        if clahe == "on":
            mask_img = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
            clahe_obj = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(32,32))
            mask_img = clahe_obj.apply(mask_img)
            mask_tensor = torch.from_numpy(mask_img).unsqueeze(0).unsqueeze(0).float() / 255.0
            print("mask shape before interpolate:", mask.shape)

        else:
            # CLAHE가 꺼져 있으면 float 그대로 유지
            mask_tensor = mask   # (1,1,H,W)
            print("mask shape before interpolate:", mask.shape)



        # 딜레이션
        if dilate_iterations > 0:
            mask_tensor = dilate_tensor(mask_tensor, kernel_size=5, iterations=dilate_iterations)


        # 블러
        blur_map = {"0":None,"1":3,"2":5,"3":7,"4":11,"5":13,"6":15}
        if blur_strength != "0":
            k = blur_map[blur_strength]
            mask_tensor = blur_tensor(mask_tensor, k)


        # Torch 변환
        mask = mask_tensor
        print("mask shape before interpolate:", mask.shape)


        # 원본 크기 맞추기
        if isinstance(image, torch.Tensor):
            h,w = image.shape[1], image.shape[2]
        elif isinstance(image, np.ndarray):
            h,w = image.shape[:2]
        else:  # PIL
            w,h = image.size
        
        print("mask shape before interpolate:", mask.shape)

        mask = F.interpolate(mask, size=(h,w), mode="bilinear", align_corners=False).float()
        print("mask shape after interpolate:", mask.shape)

        # 사용자 마스크 합성
        if user_mask is not None:
            if user_mask.shape[-2:] != mask.shape[-2:]:
                user_mask = F.interpolate(
                    user_mask.unsqueeze(0) if user_mask.ndim==2 else user_mask,
                    size=mask.shape[-2:], mode="bilinear", align_corners=False
                )
            if user_mask.ndim == 2:
                user_mask = user_mask.unsqueeze(0).unsqueeze(0)
            elif user_mask.ndim == 3 and user_mask.shape[0] == 1:
                user_mask = user_mask.unsqueeze(1)
            user_mask = user_mask.float()

            if blend_mode == "max":
                mask = torch.max(mask, user_mask)
            elif blend_mode == "average":
                mask = (mask + user_mask) / 2
            elif blend_mode == "overwrite":
                mask = torch.where(user_mask > 0.5, torch.ones_like(mask), mask)

        # 감마 보정
        mask = mask.pow(gamma)

        # 반전 옵션
        if invert == "on":
            mask = 1 - mask

        return (mask.squeeze(),)

#-------------------------


class BGRemoverMaskHumanAdv:
    classname = "BGRemoverMaskHumanAdv"
    node_id = "BGRemoverMaskHumanAdv"
    DISPLAY_NAME = "인물 중심 반자동 마스크(마스크)"
    DESCRIPTION = "U²-Net Human 기반 오토마스크 생성 후 입력마스크와 합성/보정"
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
                "gamma": ("FLOAT", {"default":0.600,"min":0.001,"max":2.000,"step":0.001, "tooltip": "감마 보정 (pow 값)"}),
            },
            "optional": {
                "clahe": (["off","on"], {"default":"off", "tooltip": "클라헤 보정"}),
                "user_mask": ("MASK", {"default":None, "tooltip": "사용자 추가 마스크"}),
                "blend_mode": (["max","average","overwrite"], {"default":"max", "tooltip": "마스크 합성 방식"}),
                "dilate_iterations": ("INT", {"default":0,"min":0,"max":10,"step":1, "tooltip": "마스크 팽창 횟수"}),
                "blur_strength": (["0","1","2","3","4","5","6"], {"default":"0", "tooltip": "블러 강도(0=없음)"})
            }
        }

    def __init__(self):
        self.model = u2net_human_model

    def execute(self, mask, stage="S1", amp_factor=1.000, invert="off", gamma=0.600,
                clahe="off", user_mask=None, blend_mode="max",
                dilate_iterations=0, blur_strength="0"):

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

        # 7. 팽창
        if dilate_iterations > 0:
            mask_tensor = dilate_tensor(mask_tensor, kernel_size=5, iterations=dilate_iterations)

        # 8. 블러
        blur_map = {"0":None,"1":3,"2":5,"3":7,"4":11,"5":13,"6":15}
        if blur_strength != "0":
            k = blur_map[blur_strength]
            mask_tensor = blur_tensor(mask_tensor, k)

        # 9. 유저마스크 합성
        if user_mask is not None:
            if user_mask.shape[-2:] != mask_tensor.shape[-2:]:
                user_mask = F.interpolate(
                    user_mask.unsqueeze(0) if user_mask.ndim==2 else user_mask,
                    size=mask_tensor.shape[-2:], mode="bilinear", align_corners=False
                )
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

        # 10. 감마 보정
        mask_tensor = mask_tensor.pow(gamma)

        # 11. 반전
        if invert == "on":
            mask_tensor = 1 - mask_tensor

        return (mask_tensor.squeeze(),)






#--------------------------
#  CLASS_MAPPINGS
#--------------------------

NODE_CLASS_MAPPINGS = {
    "BGRemover_Human": BGRemover_Human,
    "BGRemoverMaskHuman": BGRemoverMaskHuman,
    "BGRemover_HumanAdv": BGRemover_HumanAdv,
    "BGRemoverMaskHumanAdv": BGRemoverMaskHumanAdv,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "BGRemover_Human": "인물 중심 자동마스크",
    "BGRemoverMaskHuman": "인물 중심 자동마스크(마스크)",
    "BGRemover_HumanAdv": "인물 중심 반자동 마스크",
    "BGRemoverMaskHumanAdv": "인물 중심 반자동 마스크(마스크)",
}
