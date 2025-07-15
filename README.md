# AI03-Project-1
Sprint AI03 프로젝트 1번입니다

```bash
AI03-Project-1/
├── data/                    # 원본 및 전처리된 데이터
│   ├── raw/                 # 원본 데이터
│   ├── processed/           # 전처리된 데이터
│   └── external/            # 외부에서 받은 공개 데이터 등
│
├── notebooks/               # 실험/EDA용 Jupyter 노트북
│   └── 01_eda.ipynb
│
├── src/                     # 핵심 코드
│   ├── __init__.py
│   ├── dataset.py           # Custom Dataset
│   ├── model.py             # 모델 정의
│   ├── train.py             # 학습 루프
│   ├── eval.py              # 평가 로직
│   ├── config.py            # 설정 값 정리 (예: argparse or dict)
│   └── utils.py             # 공통 함수들 (seed 고정, metrics 등)
│
├── experiments/             # 실험 로그 및 결과 저장
│   ├── exp_001/             
│   │   ├── config.yaml      # 실험 설정
│   │   ├── model.pt         # 모델 가중치
│   │   └── results.json     # 결과 기록
│   └── ...
│
├── scripts/                 # 쉘스크립트, 파이프라인 자동화 등
│   └── train.sh
│
├── outputs/                 # 로그, 시각화 이미지, confusion matrix 등
│   └── figures/
│
├── requirements.txt         # 패키지 목록 (pip 기반)
├── environment.yml          # Conda 환경 파일 (선택)
├── .gitignore
└── README.md
```
