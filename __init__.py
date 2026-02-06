import logging

__version__ = "2.2.5"
logging.info(f"### Loading: ComfyUI_BGRemover (v{__version__})")

from .MODEL.BGREMOVER_Lite import (
    BGRemover_Background,
    BGRemoverMaskBackground,
    BGRemover_BackgroundAdv,
    BGRemoverMaskBackgroundAdv,
)

from .MODEL.BGREMOVER_Human import (
    BGRemover_Human,
    BGRemoverMaskHuman,
    BGRemover_HumanAdv,
    BGRemoverMaskHumanAdv,
)

from .MODEL.BGREMOVER_Util import (
    BGTileSoftFillng,
    BGmask_Subeditor,
    BGmask_CompositeAdv,
    BGRemoverMaskAmplier,
)
NODE_CLASS_MAPPINGS = {
    "BGRemover_Background": BGRemover_Background,
    "BGRemoverMaskBackground": BGRemoverMaskBackground,
    "BGRemover_BackgroundAdv": BGRemover_BackgroundAdv,
    "BGRemoverMaskBackgroundAdv": BGRemoverMaskBackgroundAdv,
    "BGRemover_Human": BGRemover_Human,
    "BGRemoverMaskHuman": BGRemoverMaskHuman,
    "BGRemover_HumanAdv": BGRemover_HumanAdv,
    "BGRemoverMaskHumanAdv": BGRemoverMaskHumanAdv,
    "BGTileSoftFillng": BGTileSoftFillng,
    "BGmask_Subeditor": BGmask_Subeditor,
    "BGmask_CompositeAdv": BGmask_CompositeAdv,
    "BGRemoverMaskAmplier": BGRemoverMaskAmplier,
}


NODE_DISPLAY_NAME_MAPPINGS = {
        "BGRemover_Background": "배경 중심 자동마스크",
        "BGRemoverMaskBackground": "배경 중심 자동마스크(마스크)",
        "BGRemover_BackgroundAdv": "배경 중심 반자동 마스크",
        "BGRemoverMaskBackgroundAdv": "배경 중심 반자동마스크(마스크)",
        "BGRemover_Human": "인물 중심 자동마스크",
        "BGRemoverMaskHuman": "인물 중심 자동마스크(마스크)",
        "BGRemover_HumanAdv": "인물 중심 반자동 마스크",
        "BGRemoverMaskHumanAdv": "인물 중심 반자동 마스크(마스크)",
        "BGTileSoftFillng": "타일링 보정채우기",
        "BGmask_Subeditor": "마스크 서브에디터",
        "BGmask_CompositeAdv": "콤포짓 마스크 어드밴스",
        "BGRemoverMaskAmplier": "마스크 증폭기",
}