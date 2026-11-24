# YOLO11을 통한 가위바위보 손동작 탐지

직접 촬영한 손 사진으로 **가위(Scissors)·바위(Rock)·보(Paper)** 데이터셋을 만들고, 사전 학습된 **YOLO11n** 모델을 파인튜닝해 동영상 속 손동작을 실시간으로 탐지하는 프로젝트입니다
전 과정은 Google Colab(Tesla T4 GPU)에서 하나의 노트북으로 진행했습니다

---

## 📌 프로젝트 개요

| 항목 | 내용 |
| --- | --- |
| 작업 | 객체 탐지(Object Detection) |
| 클래스 | `Paper`, `Rock`, `Scissors` (3개) |
| 모델 | Ultralytics YOLO11n (사전 학습 가중치 `yolo11n.pt`에서 전이 학습) |
| 데이터 | 직접 촬영·라벨링한 361장 (Roboflow에서 YOLOv11 형식으로 내보냄) |
| 환경 | Google Colab, Tesla T4, Python 3.13, Ultralytics 8.4 |
| 결과물 | 학습된 가중치(`best.pt`) + 탐지 결과 동영상(`DetectionResult.avi`) |

## 📁 폴더 구조

```
YoloObjectDetection/
├── RockScissorsPaperWithYolo.ipynb   # 전체 과정이 담긴 Colab 노트북
├── yolo11n.pt                        # 사전 학습된 YOLO11n 가중치
├── InputVideo.mp4                    # 테스트용 입력 동영상
├── DetectionResult.avi               # 탐지 결과 동영상
├── RockScissorsPaper-1/              # Roboflow에서 내려받은 데이터셋
│   ├── data.yaml
│   ├── train/ (images, labels)
│   ├── valid/ (images, labels)
│   └── test/  (images, labels)
└── runs/detect/
    ├── train/                        # 학습 로그, 그래프, 혼동 행렬
    │   └── weights/ (best.pt, last.pt)
    └── predict/                      # 동영상 추론 결과
```
