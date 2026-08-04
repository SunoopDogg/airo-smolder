#!/usr/bin/env python3
"""주석영상 생성 — fire/smoke 모델로 영상에 박스 그려 저장.

사용법:
    .venv/bin/python src/annotate_videos.py [영상경로_또는_폴더]

기본: samples/ 내 모든 mp4 처리 → runs/annotated/ 에 주석 avi 저장 + 클래스별 탐지 통계 출력.

옵션은 아래 상수로 조절.
"""
import sys
import glob
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent   # src/ 상위 = 저장소 루트

# --- 설정 ---
MODEL   = ROOT / "models" / "yolo11s-firedetect.pt"   # 채택 모델 (YOLO11s, Fire/Smoke)
CONF    = 0.4        # 신뢰도 임계 (오탐 많으면 ↑, 미탐 많으면 ↓)
IMGSZ   = 640
STRIDE  = 2          # N프레임마다 처리 (2=15fps, 속도↑). 전프레임 원하면 1
DEVICE  = 0          # GPU=0, CPU='cpu'
OUT_DIR = ROOT / "runs"


def collect(src: str):
    """입력 경로 → 영상 파일 리스트."""
    p = Path(src)
    if p.is_dir():
        vids = []
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            vids += glob.glob(str(p / ext))
        return sorted(vids)
    return [src]


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "samples")
    videos = collect(src)
    if not videos:
        print(f"영상 없음: {src}")
        sys.exit(1)

    model = YOLO(str(MODEL))
    print(f"model={MODEL.name} classes={model.names} conf={CONF} stride={STRIDE} device={DEVICE}")
    fps_eff = 30 / STRIDE

    for vid in videos:
        vn = Path(vid).name
        cnt, first_fire, proc = {}, None, 0
        results = model.predict(vid, stream=True, device=DEVICE, conf=CONF, imgsz=IMGSZ,
                                vid_stride=STRIDE, save=True, project=str(OUT_DIR),
                                name="annotated", exist_ok=True, verbose=False)
        for i, r in enumerate(results):
            proc = i + 1
            for c in r.boxes.cls:
                k = model.names[int(c)].lower()
                cnt[k] = cnt.get(k, 0) + 1
                if k == "fire" and first_fire is None:
                    first_fire = i / fps_eff
        ff = f"{first_fire:.0f}s" if first_fire is not None else "-"
        print(f"  {vn}: fire={cnt.get('fire',0)} smoke={cnt.get('smoke',0)} "
              f"first_fire@{ff} ({proc}f)")

    print(f"주석영상 저장 -> {OUT_DIR/'annotated'}")


if __name__ == "__main__":
    main()
