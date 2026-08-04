<div align="center">
  <h1>airo-smolder</h1>
  <p>ESS 차량 LFP 배터리 <b>열폭주(thermal runaway)</b> 초기 화염·연기 탐지</p>
</div>

---

## 개요

ESS 차량에 실린 LFP 리튬 배터리의 열폭주를 **화염이 붙기 전 단계**에서 잡는 것이 목표.
YOLO11s 기반 vision 탐지기로 시험영상의 fire/smoke를 프레임 단위 검출한다.

배포 타겟은 **Jetson Nano Developer Kit** (Maxwell, 4GB, JetPack 4.6).
학습·검증은 워크스테이션(RTX 4090), Nano는 ONNX → TensorRT INT8 추론만 담당한다.

## 요구사항

- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)
- NVIDIA GPU

## 설치

실행 환경은 컨테이너다. host에 Python을 따로 깔 필요 없다.

```bash
docker compose --profile gpu up -d
docker exec -it airo-smolder-gpu bash

# 이하 컨테이너 안 (/root/airo-smolder)
uv sync
```

## 사용법

컨테이너 안에서:

```bash
# samples/ 전체 처리 → runs/annotated/
.venv/bin/python src/annotate_videos.py

# 특정 영상/폴더 지정
.venv/bin/python src/annotate_videos.py samples/260422-LFP-열폭주시도1.mp4
```

출력 예:

```
model=yolo11s-firedetect.pt classes={0: 'Fire', 1: 'Smoke'} conf=0.4 stride=2 device=0
  260422-LFP-열폭주시도1.mp4: fire=2387 smoke=198 first_fire@14s (9264f)
```

임계값·stride 등은 `src/annotate_videos.py` 상단 상수로 조절한다.

## 구조

```
airo-smolder/
├── src/annotate_videos.py   주석영상 생성 (YOLO11s, conf 0.4, stride 2)
├── models/                  가중치 (git 제외)
├── samples/                 원본 시험영상 (git 제외, ~2.4GB)
└── runs/annotated/          박스 주석 출력 (git 제외)
```
