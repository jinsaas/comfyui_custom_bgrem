####bgrem V.2.25####
이 프로젝트는 comfyui092 이후 및 최신버전에서도 동작합니다.

####변경사항####
타일링 채우기 기능의 버그를 정리했습니다. 
간격 오버하는 상황시 자동으로 잘라내어 픽셀 불일치 오류 가능성이 줄었습니다.

####bgrem은 다음 원칙을 기반으로 설계되었습니다####
• 환경 안정성 최우선
• 결정성 100% (Portable 환경에서도 동일 결과)
• 유지보수 가능한 구조
•Stable Diffusion 모델 기반을 사용하지 않는 ComfyUI Standalone 환경에서의 
 안정성을 무너뜨리지 않는 형태를 지향합니다.

 
# Original U²Net implementation:
 https://github.com/xuebinqin/U-2-Net
# Paper: U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection (Pattern Recognition, 2020)
----------
## Licenses
- Original U²Net implementation: MIT License (see external_licenses/U2Net-MIT.txt)Apache License 2.0 (see external_licenses/Apache-2.0.txt)
- Cloth Segmentation model: openrail
- Converted versions: safetensors format, for efficiency in use

“원본은 Apache License 2.0을 따릅니다. 변환본은 safetensors로 제작되었습니다.”
사실상 u2Net의 코드를 직접 참고하고 있으므로 저는 라이선스를 놔두지 않았고, 
아파치 라이선스 전문을 포함한 외부 라이선스는 모두 exterlal licenses에 들어가 있습니다.
----------
## Pretrained Models Used
- u2net.pth (salient object detection, general)
- u2net_human_seg.pth (human segmentation)
- u2net_portrait.pth (portrait segmentation)
-----------
# Source:
# https://github.com/xuebinqin/U-2-Net

-기본적인 소스와 모델은 여기서 가져왔습니다.

해당 PTH는 모두 용량 절감을 위해 개인사용중인 변환기로 세이프텐서로 변환해 사용 중입니다.
------------
# Original U²Net recommended requirements (reference only)
------------
numpy
scikit-image
torch
torchvision
pillow
opencv-python
gradio
gradio_client
httpx
opt_einsum
colorlog
rarfile
paddlepaddle
paddlehub
paddlenlp
aistudio-sdk
decorator

--------------
참고 사항으로 가져왔습니다. 
이 리스트는 원본 U²Net 레포지토리에서 권장된 의존성입니다. 
BGRemover는 Torch 기반으로 동작하므로 실제 필수 의존성은 최소 리스트만 필요합니다.

###requirements.txt가 비어 있는 이유###
ComfyUI는 확장팩 로딩 시 requirements.txt를 자동 설치합니다.
그러나 pip나 git을 통한 원격설치 방식은 환경 손상 위험이 매우 높기 때문에,
requirements.txt를 빈 파일로 유지합니다.

대신, 다음의 코드로 의존성 설치법을 남깁니다:
의존성 풀 인스톨 리스트
(U-Net 원본의 의존성 요구 사항입니다. 
 BGRemover는 원본급의 의존성이 아니라 이 설치법방법 이후에 있는 최소한의 의존성만 채워도 동작합니다.)
 
python.exe -m pip install --no-deps numpy
python.exe -m pip install --no-deps scikit-image
python.exe -m pip install --no-deps torch
python.exe -m pip install --no-deps torchvision
python.exe -m pip install --no-deps pillow
python.exe -m pip install --no-deps opencv-python
python.exe -m pip install --no-deps gradio
python.exe -m pip install --no-deps gradio_client
python.exe -m pip install --no-deps httpx
python.exe -m pip install --no-deps opt_einsum
python.exe -m pip install --no-deps colorlog
python.exe -m pip install --no-deps rarfile
python.exe -m pip install --no-deps aistudio-sdk==0.2.6
python.exe -m pip install --no-deps paddlenlp==2.6.1   
python.exe -m pip install --no-deps paddlepaddle==2.6.2
python.exe -m pip install --no-deps paddlehub==2.4.0
python.exe -m pip install --no-deps decorator

-
 1.의존성 안정 조합(BG Remover계열 전용)
 aistudio-sdk==0.2.6
 paddlenlp==2.6.1
 paddlehub==2.3.x ~ 2.4.x
 paddlepaddle==2.5.x ~ 2.6.x
 protobuf==3.20.3
 
-
2.타협안 (최신 유지 + 환경 변수 설정) (코더용)
최신 protobuf 유지
실행 전에 환경 변수 설정:
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
최신 보안 정책을 유지하면서 충돌을 회피.(성능은 다소 떨어질 수 있음.)
-
3.protobuf버전 유지로 해결하는 법 (코더/전문개발자용/업데이트시 재수행이 필요할 수 있음)
- Paddle 내부의 .proto 파일을 최신 protoc >= 3.19.0으로 다시 컴파일하면 
 Descriptor 생성 방식이 최신 규칙에 맞게 바뀝니다. 
 (혹은 paddle 패키지 내의 proto 폴더의 init파일 외 5개의 py파일을 싹 재수정. 
 수정작업이 파일당 7만자 이상이라 전문개발자용이라는 이유가 여기 있습니다.)
- 하지만 ComfyUI 내장 파이썬 환경에서 이 작업을 하려면 
protoc 설치 + Paddle 소스 수정이 필요해서 일반 사용자에게는 부담이 큽니다.

- protoc 설치
- 최신 버전(≥ 3.19.0)을 설치해야 합니다.
- Windows라면 공식 릴리즈 페이지 (github.com in Bing)에서 
  protoc-<버전>-win64.zip 다운로드 후 PATH에 추가.
- Paddle 소스 코드 준비
- 현재 ComfyUI 내장 파이썬에는 wheel 패키지(.whl)로 설치된 Paddle이 있음.
- .proto 파일은 Paddle 소스 저장소(PaddlePaddle GitHub)나 wheel 내부에 포함.
- 예: Lib\site-packages\paddle\base\proto\*.proto
- proto 파일 재컴파일
- 해당 디렉토리로 이동해서 protoc 실행:
protoc --python_out=. data_feed.proto
protoc --python_out=. trainer_desc.proto
protoc --python_out=. attribute.proto
- 이렇게 하면 .proto 파일이 최신 규칙에 맞는 _pb2.py 파일로 다시 생성.
- 기존 파일 교체
- 새로 생성된 _pb2.py 파일을 paddle\base\proto 디렉토리에 덮어쓰기.
- ComfyUI 내장 파이썬은 Lib\site-packages\paddle\base\proto 경로에 있음.
- 테스트
- ComfyUI 실행 → BGRemover 로드 → protobuf Descriptor 에러가 사라지는지 확인.

  주의할 점
Paddle wheel 패키지를 직접 수정하는 거라, 업데이트나 재설치 시 덮어씌워질 수 있음.
잘못된 proto 재생성은 Paddle 내부 API와 불일치할 수 있어, 안정성에 위험이 있음.

----

#또는 최소 의존성 설치법을 따르세요.
#----------------
#의존성 필수 리스트
#----------------
numpy
pillow
opencv-python
torch
torchvision
lazy_loader
tifffile
scipy
scikit-image
#----------------
#----------------------------------------------------

python.exe -m pip install --no-deps numpy(포터블은 할 필요가 없음.)
python.exe -m pip install --no-deps pillow
python.exe -m pip install --no-deps opencv-python
python.exe -m pip install --no-deps torch
python.exe -m pip install --no-deps torchvision
python.exe -m pip install --no-deps lazy_loader
python.exe -m pip install --no-deps tifffile
python.exe -m pip install --no-deps scipy
python.exe -m pip install --no-deps scikit-image

#----------------------------------------------------
Comfyui 기본 설치에 해당하는 torch, torchvision, torchAudio는 당연히 설치했을거라 제외했습니다.
위 패키지들은 BGRemover에서 주로 사용하는 안전한 최소 의존성입니다.
로드시 작동 불가가 나오는 것만 추가설치하세요.
위 의존성 이후의 추가설치는 선택 사항이며, 대부분의 기능은 기본 상태에서도 작동합니다.



[설치 방법]



Installation:

1. ZIP 다운로드: 저장소를 .zip 파일로 다운로드합니다.

2. custom_nodes 폴더에 배치: 압축을 풀고 ComfyUI/custom_nodes 폴더에 넣습니다.

3. ComfyUI 재시작: 재시작하면 노드가 로드됩니다.


####bgrem Node Usage####

[Auto Nodes]

BGRemover_Background          배경 중심 자동마스크
- node_id: BGRemover_Background
- category: 리무버/자동
- 역할: 이미지에서 배경바탕으로 마스크 자동선택. 인버트로 지정변환 가능.
- Inputs: image 입력 이미지
- Inputs: stage 사용할 스테이지 선택
- Inputs: 증폭검출
- Inputs: invert 선택반전
- Inputs: gamma 감마조정
- Inputs: clahe 클라헤 보정
- Outputs: mask

BGRemoverMaskBackground       배경 중심 자동마스크(마스크입력)
- node_id: BGRemoverMaskBackground
- category: 리무버/자동
- 역할: 마스크에서 배경바탕으로 마스크 자동선택. 인버트로 지정변환 가능.
- Inputs: Mask 입력 마스크
- Inputs: stage 사용할 스테이지 선택
- Inputs: 증폭검출
- Inputs: invert 선택반전
- Inputs: gamma 감마조정
- Inputs: clahe 클라헤 보정
- Outputs: mask

BGRemover_Human               인물 중심 자동마스크
- node_id: BGRemover_Human
- category: 리무버/자동
- 역할: 이미지에서 사람 기준으로 마스크 자동선택. 인버트로 지정변환 가능.
- Inputs: image 입력 이미지
- Inputs: stage 사용할 스테이지 선택
- Inputs: 증폭검출
- Inputs: invert 선택반전
- Inputs: gamma 감마조정
- Inputs: clahe 클라헤 보정
- Outputs: mask 마스크출력

BGRemoverMaskHuman            인물 중심 자동마스크(마스크입력)
- node_id: BGRemoverMaskHuman
- category: 리무버/자동
- 역할: 마스크에서 사람 기준으로 마스크 자동선택. 인버트로 지정변환 가능.
- Inputs: Mask 입력 마스크
- Inputs: stage 사용할 스테이지 선택
- Inputs: 증폭검출
- Inputs: invert 선택반전
- Inputs: gamma 감마조정
- Inputs: clahe 클라헤 보정
- Outputs: mask 마스크출력

[SemiAuto Nodes]

BGRemover_BackgroundAdv          배경 중심 반자동마스크
- node_id: BGRemover_BackgroundAdv
- category: 리무버/반자동
- 역할: 이미지에서 배경바탕으로 마스크 자동선택+유저 마스크 주입 가능. 인버트로 지정변환 가능.
- Inputs: image 입력 이미지
- Inputs: stage 사용할 스테이지 선택
- Inputs: 증폭검출
- Inputs: invert 선택반전
- Inputs: gamma 감마조정
- Inputs: clahe 클라헤 보정
- Inputs: (선택)mask 사용자 추가 마스크
- Inputs: (선택)blend_mode 마스크 합성 방식
- Outputs: mask 마스크출력

BGRemoverMaskBackgroundAdv       배경 중심 반자동마스크(마스크)
- node_id: BGRemoverMaskBackgroundAdv
- category: 리무버/반자동
- 역할: 이미지에서 배경바탕으로 마스크 자동선택+유저 마스크 주입 가능. 인버트로 지정변환 가능.
- Inputs: Mask 입력 마스크
- Inputs: stage 사용할 스테이지 선택
- Inputs: 증폭검출
- Inputs: invert 선택반전
- Inputs: gamma 감마조정
- Inputs: clahe 클라헤 보정
- Inputs: (선택)mask 사용자 추가 마스크
- Inputs: (선택)blend_mode 마스크 합성 방식
- Outputs: mask 마스크출력

BGRemover_HumanAdv              인물 중심 반자동 마스크
- node_id: BGRemover_HumanAdv
- category: 리무버/반자동
- 역할: 이미지에서 사람 기준으로 자동선택+유저 마스크 주입 가능. 인버트로 지정변환 가능.
- Inputs: image 입력 이미지
- Inputs: stage 사용할 스테이지 선택
- Inputs: 증폭검출
- Inputs: invert 선택반전
- Inputs: gamma 감마조정
- Inputs: clahe 클라헤 보정
- Inputs: (선택)mask 사용자 추가 마스크
- Inputs: (선택)blend_mode 마스크 합성 방식
- Outputs: mask 마스크출력

BGRemoverMaskHumanAdv           인물 중심 반자동 마스크(마스크)
- node_id: BGRemoverMaskHumanAdv
- category: 리무버/반자동
- 역할: 이미지에서 사람 기준으로 자동선택+유저 마스크 주입 가능. 인버트로 지정변환 가능.
- Inputs: mask 입력 이미지
- Inputs: stage 사용할 스테이지 선택
- Inputs: 증폭검출
- Inputs: invert 선택반전
- Inputs: gamma 감마조정
- Inputs: clahe 클라헤 보정
- Inputs: (선택)mask 사용자 추가 마스크
- Inputs: (선택)blend_mode 마스크 합성 방식
- Outputs: mask 마스크출력

[Processing Nodes]


BGTileSoftFillng              타일링 보정 채우기
- node_id: BGTileSoftFillng
- category: 리무버/유틸
- 역할: pytorch기반 타일링 채우기. 부분이미지 보정용으로 쓸 수 있습니다.
- Inputs: image 미리 잘라둔 이미지
- canvas_width: 캔버스 X축 크기만큼 비례 타일링할 픽셀깊이
- canvas_height: 캔버스 Y축 크기만큼 비례 타일링할 픽셀깊이
- rematch:타일화 처리의 경계면을 덮어 위화감을 보정할 중복 겹치기 횟수
- Outputs: image

BGMasksubeditor                마스크 서브에디터
- node_id: BGMasksubeditor
- category: 리무버/유틸
- 역할: pytorch기반 마스크 지우개. 지울영역만큼의 수정한 마스크를 넣고 빼는 것으로 마스크 부분 보정용으로 쓸 수 있습니다.
- Inputs: mask 작업 수정할 마스크
- Inputs: mask 지울 영역을 그린 마스크
- subtract_mode:제거 방식. 완전제거-부분제거-차이값만 남김 (유저마스크 혹은 베이스가 넘은부분은 그대로 적용)이 있습니다.
- blur_mode:블러 셋팅. 블러 미적용, 소프트와 하드가 있습니다.
- blur_strength:블러 커널 크기
- add_mask:(선택)덧붙일 추가 마스크
- Outputs: mask

BGmask_CompositeAdv            콤포짓 마스크 어드밴스
- node_id: BGmask_CompositeAdv
- category: 리무버/유틸
- 역할: pytorch기반 마스크 지우개. 지울영역만큼의 수정한 마스크를 넣고 빼는 것으로 마스크 부분 보정용으로 쓸 수 있습니다.
- Inputs: mask 작업 수정할 마스크
- Inputs: mask 지울 영역을 그린 마스크
- subtract_mode:제거 방식. 완전제거-부분제거-차이값만 남김 (유저마스크 혹은 베이스가 넘은부분은 그대로 적용)이 있습니다.
- blur_mode:블러 셋팅. 블러 미적용, 소프트와 하드가 있습니다.
- blur_strength:블러 커널 크기
- add_mask:(선택)덧붙일 추가 마스크
- add_mask:(선택)덧붙일 추가 마스크
- add_mask:(선택)덧붙일 추가 마스크
- add_mask:(선택)덧붙일 추가 마스크
- add_mask:(선택)덧붙일 추가 마스크
- Outputs: mask

BGRemoverMaskAmplier          마스크 증폭기
- node_id: BGRemoverMaskAmplier
- category: 리무버/유틸
- 역할: 마스크를 입력해서 부분증폭 및 전체증폭, 추가처리를 합니다.
- Inputs: mask 입력 마스크
- Inputs: 증폭검출
- Inputs: invert 선택반전
- Inputs: gamma 감마조정
- Inputs: clahe 클라헤 보정
- Inputs: (선택)mask 사용자 추가 마스크
- Inputs: (선택)blend_mode 마스크 합성 방식
- Inputs: (선택)부분강조
- Inputs: (선택)부분강조 강도
- Outputs: mask 마스크출력