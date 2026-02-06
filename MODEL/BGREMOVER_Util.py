#--------------------------
#  BG_Remover Node Header 
#--------------------------

import torch
import random
import numpy as np
import torch.nn.functional as F
import cv2

#--------------------------
#  Utillity Header
#--------------------------

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

def normalize_mask(mask_tensor: torch.Tensor) -> torch.Tensor:
    return (mask_tensor - mask_tensor.min()) / (mask_tensor.max() - mask_tensor.min() + 1e-8)

 
 
def make_feather_mask(h, w, c, border=16):
    mask = torch.ones((h,w,c), dtype=torch.float32)
    for k in range(border):
        alpha = k / border
        mask[k,:,:] *= alpha
        mask[-k-1,:,:] *= alpha
        mask[:,k,:] *= alpha
        mask[:,-k-1,:] *= alpha
    return mask


def subtract_mask(base, *removes):
    # 여러 제거 마스크를 동시에 적용
    combined_remove = np.maximum.reduce(removes)
    return np.clip(base - combined_remove, 0, 1)

def difference_mask(mask_a, mask_b):
    return np.abs(mask_a - mask_b)


#--------------------------
#  BG_Remover Utillity Node
#--------------------------

class BGTileSoftFillng:
    classname = "BGTileSoftFillng"
    node_id = "BGTileSoftFillng"
    DISPLAY_NAME = "타일링 보정채우기"
    DESCRIPTION = "pytorch기반 타일링 채우기. 부분이미지 보정용으로 쓸 수 있습니다."
    CATEGORY = "리무버/유틸"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_image": ("IMAGE", {"tooltip": "크롭된 타일 이미지"}),
                "canvas_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8, "tooltip": "캔버스 X축 크기"}),
                "canvas_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8, "tooltip": "캔버스 Y축 크기"}),
                "rematch": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": "경계 보정 횟수"}),
            }
        }
    def execute(self, tile_image, canvas_width, canvas_height, rematch):
        tile_image = tile_image.float()
        B, H, W, C = tile_image.shape

        # stride 계산
        rematch = min(rematch, min(H, W) - 1)
        stride_x = max(W - rematch, 1)
        stride_y = max(H - rematch, 1)

        repeat_x = canvas_width // stride_x
        repeat_y = canvas_height // stride_y

        # 캔버스 크기 고정
        canvas = torch.zeros((B, canvas_height, canvas_width, C), dtype=torch.float32)
        weight = torch.zeros_like(canvas)

        # feather mask 생성
        mask = make_feather_mask(H, W, C, border=min(rematch, 16))

        # 타일 배치 + 경계 보정
        for i in range(repeat_y + 1):
            for j in range(repeat_x + 1):
                y0 = i * stride_y
                x0 = j * stride_x
                y1 = min(y0 + H, canvas_height)
                x1 = min(x0 + W, canvas_width)

                h_slice = y1 - y0
                w_slice = x1 - x0

                if h_slice > 0 and w_slice > 0:
                    # 기본 타일 배치
                    canvas[:, y0:y1, x0:x1, :] += tile_image[:, :h_slice, :w_slice, :]
                    weight[:, y0:y1, x0:x1, :] += 1.0

                    # 경계 보정용 추가 배치
                    canvas[:, y0:y1, x0:x1, :] += tile_image[:, :h_slice, :w_slice, :] * mask[:h_slice, :w_slice, :]
                    weight[:, y0:y1, x0:x1, :] += mask[:h_slice, :w_slice, :]

        # 평균화 및 빈칸 처리
        canvas = canvas / torch.clamp(weight, min=1.0)
        canvas[weight == 0] = 0

        return (canvas,)



#--------------------------

class BGmask_Subeditor:
    classname = "BGmask_Subeditor"
    node_id = "BGmask_Subeditor"
    DISPLAY_NAME = "마스크 서브에디터"
    DESCRIPTION = "마스크 추가 작업용 에디터. 과처리된 마스크를 손질하거나 덧붙일 수 있습니다."
    CATEGORY = "리무버/유틸"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_mask": ("MASK", {"tooltip": "작업할 마스크"}),
                "user_mask": ("MASK", {"tooltip": "지울 영역 선택"}),
                "subtract_mode": (["delete", "subtract", "difference"], {"default": "delete", "tooltip": "제거 방식.\n 완전제거-부분제거-차이값만 남김\n(유저마스크 혹은 베이스가 넘은부분은 그대로 적용)이 있습니다."}),
                "blur_mode": (["none","soft", "hard"], {"default": "none", "tooltip": "블러 셋팅"}),
                "blur_strength": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10, "step": 1, "tooltip": "블러 커널 크기"})
            },
            "optional": {
                "add_mask": ("MASK", {"default": None, "tooltip": "덧붙일 추가 마스크"})
            }
        }
    def execute(self, base_mask, user_mask, subtract_mode, blur_mode, blur_strength=0, add_mask=None):
        result = base_mask.clone()

        # 제거 모드 처리
        if subtract_mode == "delete":
            result = torch.where(user_mask > 0.5, torch.zeros_like(result), result)
        elif subtract_mode == "subtract":
            result = torch.clamp(result - user_mask, min=0.0, max=1.0)
        elif subtract_mode == "difference":
            result = torch.abs(result - user_mask)

        # 추가 마스크 합성
        if add_mask is not None:
            result = torch.clamp(result + add_mask, min=0.0, max=1.0)
            
        # 차원 보정
        if result.dim() == 2:        # (H, W)
            result = result.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif result.dim() == 3:      # (H, W, C) 또는 (C,H,W)
            result = result.unsqueeze(0)               # (1,C,H,W)
        elif result.dim() == 4:      # 이미 (N,C,H,W)
            pass                     # 그대로 사용

            
        # 블러 처리
        if blur_strength > 0 and blur_mode != "none":
            k = min(2 * int(blur_strength) - 1, 11)  # 최대 11x11 제한
            kernel = torch.ones((1,1,k,k), device=result.device) / (k*k)

            if blur_mode == "soft":
                result = F.conv2d(result, kernel, padding=k//2)
            elif blur_mode == "hard":
                # hard 모드: 커널 크기를 조금 더 키워서 한 번만 적용
                k_hard = min(2 * int(blur_strength) + 1, 13)
                kernel_hard = torch.ones((1,1,k_hard,k_hard), device=result.device) / (k_hard*k_hard)
                result = F.conv2d(result, kernel_hard, padding=k_hard//2)
        else:
            pass


            
        # 차원 보정

                 
        if result.dim() == 4: 
            result = result.squeeze(0).squeeze(0) 
        elif result.dim() == 3:  
            result = result.squeeze(0)
        elif result.dim() == 2:  
            pass  
            
        return (result,)

#--------------------------



class BGmask_CompositeAdv:
    classname = "BGmask_CompositeAdv"
    node_id = "BGmask_CompositeAdv"
    DISPLAY_NAME = "콤포짓 마스크 어드밴스"
    DESCRIPTION = "다중 마스크 합성 + 경계보정 + 팽창/축소(소프트) + 블러 처리."
    CATEGORY = "리무버/유틸"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_mask": ("MASK", {"tooltip": "원본 마스크"}),
                "user_mask": ("MASK", {"tooltip": "합성 대상 마스크"}),
                "blend_mode": (["max","average","overwrite", "or", "and"], {"default":"max", "tooltip": "합성 방식 지정"}),
                "feather_strength": ("FLOAT", {"default":0,"min":0,"max":5,"step":1, "tooltip":"페더링 강도 설정"}),
                "mask_adjust_unit": ("INT", {"default":0,"min":-5,"max":5,"step":1,
                                             "tooltip":"양수=팽창, 음수=축소 (소프트 처리)"}),
                "blur_mode": (["none","soft","hard"], {"default":"none", "tooltip": "블러 셋팅"}),
                "blur_strength": ("INT", {"default":0,"min":0,"max":10,"step":1, "tooltip":"블러 강도 설정"}),
                "invert": (["off","on"], {"default":"off", "tooltip": "반전"}),
            },
            "optional": {
                "add_mask1": ("MASK", {"default":None, "tooltip":"추가 마스크"}),
                "add_mask2": ("MASK", {"default":None, "tooltip":"추가 마스크"}),
                "add_mask3": ("MASK", {"default":None, "tooltip":"추가 마스크"}),
                "add_mask4": ("MASK", {"default":None, "tooltip":"추가 마스크"}),
                "add_mask5": ("MASK", {"default":None, "tooltip":"추가 마스크"}),
            }
        }

    def execute(self, base_mask, user_mask, blend_mode="max",
                feather_strength=1, mask_adjust_unit=0,
                blur_mode="none", blur_strength=0, invert="off",
                add_mask1=None, add_mask2=None, add_mask3=None,
                add_mask4=None, add_mask5=None):

        # 차원 보정
        result = ensure_mask_tensor(base_mask)
        user_mask = ensure_mask_tensor(user_mask)


        # 유저마스크 합성
        if blend_mode == "max":
            result = torch.max(result, user_mask)
        elif blend_mode == "average":
            result = (result + user_mask) / 2
        elif blend_mode == "overwrite":
            result = torch.where(user_mask > 0.5, torch.ones_like(result), result)
        elif blend_mode == "or":
            result = torch.clamp(result + user_mask, 0, 1)
        elif blend_mode == "and":
            result = result * user_mask

        # 추가 마스크 합성
        for add_mask in [add_mask1, add_mask2, add_mask3, add_mask4, add_mask5]:
            if add_mask is not None:
                add_mask = ensure_mask_tensor(add_mask)
                if blend_mode == "max":
                    result = torch.max(result, add_mask)
                elif blend_mode == "average":
                    result = (result + add_mask) / 2
                elif blend_mode == "overwrite":
                    result = torch.where(add_mask > 0.5, torch.ones_like(result), result)
                elif blend_mode == "or":
                    result = torch.clamp(result + add_mask, 0, 1)
                elif blend_mode == "and":
                    result = result * add_mask



        # 경계 보정 (페더링: 커널 기반 블러)
        if feather_strength > 0:
            k = 2 * int(feather_strength) - 1  # 홀수 커널 크기
            
            kernel = torch.ones((1,1,k,k), device=result.device) / (k*k)
            result = F.conv2d(result, kernel, padding=k//2)
        else:
            # feather_strength == 0 → 페더링 스킵
            pass

        # 팽창/축소 (소프트 처리)
        if mask_adjust_unit != 0:
            k = min(2*abs(mask_adjust_unit)+1, 9)  # 최대 9x9 제한

            kernel = torch.ones((1,1,k,k), device=result.device) / (k*k)
            result = F.conv2d(result, kernel, padding=k//2)
            
            if mask_adjust_unit > 0:  # 팽창
                result = torch.clamp(result, 0, 1)
            
            elif mask_adjust_unit < 0:  # 축소
                result = torch.sigmoid(5*(result-0.5))
                
            else:  
                # mask_adjust_unit == 0 → 스킵

                pass

        # 블러 처리
        if blur_strength > 0 and blur_mode != "none":
            k = 2 * blur_strength - 1
            kernel = torch.ones((1,1,k,k), device=result.device) / (k*k)
            if blur_mode == "soft":
                result = F.conv2d(result, kernel, padding=k//2)
            elif blur_mode == "hard":
                result = F.conv2d(result, kernel, padding=k//2)
                result = F.conv2d(result, kernel, padding=k//2)
            else:
                # blur_strength == 0 또는 blur_mode == "none" → 블러 스킵
            
                pass
                
        # 반전 옵션
        if invert == "on":
            result = 1 - result


        # 최종 출력
        result = torch.clamp(result,0,1)
        return (result.squeeze(),)

#--------------------------



class BGRemoverMaskAmplier:
    classname = "BGRemoverMaskAmplier"
    node_id = "BGRemoverMaskAmplier"
    DISPLAY_NAME = "마스크 증폭기"
    DESCRIPTION = "입력 마스크를 기반으로 증폭/보정 후 유저마스크와 합성 + 부분강조"
    CATEGORY = "리무버/유틸"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "입력 마스크"}),
                "amp_factor": ("FLOAT", {"default":1.000,"min":1.000,"max":2.000,"step":0.001, "tooltip": "출력 증폭 계수"}),
                "invert": (["off","on"], {"default":"off", "tooltip": "반전"}),
                "gamma": ("FLOAT", {"default":0.600,"min":0.001,"max":2.000,"step":0.001, "tooltip": "감마 보정 (pow 값)"}),
            },
            "optional": {
                "clahe": (["off","on"], {"default":"off", "tooltip": "클라헤 보정"}),
                "user_mask": ("MASK", {"default":None, "tooltip": "합성 대상 마스크"}),
                "blend_mode": (["max","average","overwrite"], {"default":"max", "tooltip": "합성 방식 지정"}),
                "highlight_region": (["none","left","right","top","bottom"], {"default":"none", "tooltip":"부분 강조 방향"}),
                "highlight_strength": ("FLOAT", {"default":0.0,"min":0.0,"max":2.0,"step":0.1, "tooltip":"부분 강조 강도"})
            }
        }

    def execute(self, mask, amp_factor=1.000, invert="off", gamma=0.600,
                clahe="off", user_mask=None, blend_mode="max",
                highlight_region="none", highlight_strength=0.0):

        mask_tensor = mask.unsqueeze(0).unsqueeze(0).float()
        mask_tensor = torch.clamp(mask_tensor * amp_factor, 0, 1)

        # CLAHE
        if clahe == "on":
            mask_img = (mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
            clahe_obj = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(32,32))
            mask_img = clahe_obj.apply(mask_img)
            mask_tensor = torch.from_numpy(mask_img).unsqueeze(0).unsqueeze(0).float() / 255.0

        # 유저마스크 합성
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

        # 부분 강조
        if highlight_region != "none" and highlight_strength > 0.0:
            B, C, H, W = mask_tensor.shape
            if highlight_region == "left":
                grad = torch.linspace(highlight_strength, 1.0, W, device=mask_tensor.device)
                grad = grad.unsqueeze(0).unsqueeze(0).unsqueeze(2).repeat(B,1,H,1)
            elif highlight_region == "right":
                grad = torch.linspace(1.0, highlight_strength, W, device=mask_tensor.device)
                grad = grad.unsqueeze(0).unsqueeze(0).unsqueeze(2).repeat(B,1,H,1)
            elif highlight_region == "top":
                grad = torch.linspace(highlight_strength, 1.0, H, device=mask_tensor.device)
                grad = grad.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(B,1,1,W)
            elif highlight_region == "bottom":
                grad = torch.linspace(1.0, highlight_strength, H, device=mask_tensor.device)
                grad = grad.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(B,1,1,W)
            mask_tensor = torch.clamp(mask_tensor * grad, 0, 1)

        # 감마 보정
        mask_tensor = mask_tensor.pow(gamma)

        # 반전
        if invert == "on":
            mask_tensor = 1 - mask_tensor

        return (mask_tensor.squeeze(),)



#--------------------------
#  CLASS_MAPPINGS
#--------------------------

NODE_CLASS_MAPPINGS = {
    "BGTileSoftFillng": BGTileSoftFillng,
    "BGmask_Subeditor": BGmask_Subeditor,
    "BGmask_CompositeAdv": BGmask_CompositeAdv,
    "BGRemoverMaskAmplier": BGRemoverMaskAmplier,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "BGTileSoftFillng": "타일링 보정채우기",
    "BGmask_Subeditor": "마스크 서브에디터",
    "BGmask_CompositeAdv": "콤포짓 마스크 어드밴스",
    "BGRemoverMaskAmplier": "마스크 증폭기",
}
