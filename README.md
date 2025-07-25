# 🟣 AI03-Project-1
- Sprint AI03 프로젝트#1 레포입니다.

## ⚪ Intro
- 프로젝트 기간:  7/15(화) ~ 7/31(목)
- 최종 발표: 7/31(목) 
- [프로젝트1 가이드](https://codeit.notion.site/AI-1b36fd228e8d80bd9f6dc99b409f952c)


## ⚪ 프로젝트 구조 (25.07.23 updated)
```bash
AI03-Project-1/
├── data/                    # 원본 및 전처리된 데이터
│   ├── raw/                 # 원본 데이터 [Ai Hub 경구약제 이미지 데이터] 경로
│   ├── processed/           # 전처리된 데이터(label_map.json, train_df.csv 등)
│   └── external/            # 외부에서 받은 공개 데이터 등 x
│
├── notebooks/               # 실험/EDA용 Jupyter 노트북 ⭐
│   ├── BASELINE.ipynb             # 실험에 사용될 팀공식 Baseline 코드
│   └── SOOHYUN_ooooo_250721.ipynb         #  개인 작업물(ex. 7월21에 작업한 ooooo.ipynb)

│
├── src/                     # 핵심 코드 ⭐ 각자 역할별로 업데이트 해야하는 소스코드
│   ├── assets/               # data에 관련된 대부분의 코드를 담당하게 됨 (은영) 
│   │   ├── __init__.py
│   │   ├── dataset.py        # PillDataset 클래스
│   │   ├── prepare_data.py   # Raw -> Processed 데이터 변환 스크립트
│   │   ├── uitls.py          # dataset 관련 utils
│   │   └── transforms.py     # Albumentations 기반 데이터 Augmentation
│   ├── model.py              # 모델 정의 (계승, 진석)
│   ├── train.py              # 학습 루프 (수현)
│   ├── eval.py               # 평가 로직 (수현)
│   ├── ensemble.py           # 최종 모델 앙상블
│   ├── config.py             # 설정 값 정리 (예: argparse or dict)
│   ├── submit.py             # 캐글 제출용 csv 파일
│   └── utils.py              # 공통 함수들 (seed 고정, metrics 등)
│
├── experiments/             # 실험 로그 및 결과 저장
│   ├── exp_250714/          #               
│   │   ├── config.yaml      # 실험 설정
│   │   ├── model.pt         # 모델 파라미터 저장값 ⭐ 구글드라이브로 공유
│   │   └── results.json     # 결과 기록 등
│   └── ...
│
├── scripts/                 # 쉘스크립트, 파이프라인 자동화 등
│   └── train.sh
│
├── output/                 # 캐글 Submission 파일(.csv)⭐, 로그, 시각화 이미지, confusion matrix 등
│   └── figures/
│
├── requirements.txt         # 패키지 목록 (pip 기반)
├── environment.yml          # Conda 환경 파일 
├── .gitignore
├── ProjectLog.md            # 프로젝트1 개발 일지 작성⭐
└── README.md                # 프로젝트 구조 업데이트
```


## ⚪ 협업 워크플로우 (Collaborator 모델)

**`main` 브랜치에는 절대로 직접 `push` 하지 않는다.** 모든 코드는 `Pull Request`를 통해서만 `main`에 합쳐진다.

-----

### **Step 1: 작업 시작 전, 항상 최신 상태로 동기화**

내 로컬 `main` 브랜치를 GitHub에 있는 최신 버전으로 업데이트하는 과정임. 다른 팀원이 작업한 내용이 반영되어 있을 수 있으니, 새로운 작업을 시작하기 전에는 반드시 해줘야 함.

```bash
# 1. main 브랜치로 이동
git checkout main

# 2. 원격 저장소(origin)의 최신 내용을 가져와서 내 로컬 main에 병합
git pull origin main
```

### **Step 2: 나만의 작업 공간(Branch) 만들기**

이제 내가 맡은 기능을 개발할 차례. `main`에서 직접 작업하면 절대 안 되고 내 작업 전용 브랜치를 새로 만들어야 함.


```bash
# 'feature/new-model' 이라는 이름의 새 브랜치를 만들고 그 브랜치로 바로 이동
git checkout -b feature/new-model
```

### **Step 3: 열심히 코딩하고, 작업 내용 기록하기 (Commit)**

  - **커밋 메시지 Tip:** "코드 수정"처럼 애매하게 쓰지 말고, "Feat: 로그인 API 추가", "Fix: 이메일 유효성 검사 오류 수정" 처럼 **무엇을** 했는지 명확하게 쓰는 게 좋음.

<!-- end list -->

```bash
# 1. 변경된 모든 파일을 Staging Area로 추가
git add .

# 2. 변경 내용을 메시지와 함께 기록(커밋)
git commit -m "Feat: 새로운 data processing method 추가"
```

### **Step 4: 내 작업을 팀원들에게 공유하기 (Push)**

로컬 컴퓨터에서 작업하고 커밋한 내용을 이제 중앙 저장소(GitHub)로 올려서 다른 팀원들도 볼 수 있게 공유

```bash
# 내가 만든 'feature/new-model' 브랜치를 원격 저장소(origin)에 올린다
git push origin feature/new-model
```

### **Step 5: 내 코드를 합쳐달라고 요청하기 (Pull Request)**

가장 중요한 단계\! GitHub 저장소 페이지로 가서, 내가 방금 `push`한 브랜치를 `main` 브랜치에 합쳐달라는 \*\*Pull Request(PR)\*\*를 생성.

  - **PR 설명 Tip:** 내가 왜 이 작업을 했고, 어떤 변경사항이 있는지, 팀원이 어떤 부분을 중점적으로 봐줬으면 하는지 상세하게 작성하면 팀원들의 시간을 아껴줄 수 있음.

### **Step 6: 함께 검토하고, 안전하게 병합하기 (Code Review & Merge)**

  - **리뷰어:** 다른 팀원들은 PR을 보고 코드에 대한 의견을 남겨. "이 부분은 이렇게 바꾸는 게 더 효율적일 것 같아요\!" 같은 피드백을 주고받음.
  - **작성자:** 피드백을 받으면 코드를 수정하고 추가로 커밋 & 푸시하면 PR에 자동으로 반영됨.
  - **병합(Merge):** 모든 리뷰가 끝나고 팀원들의 승인(Approve)을 받으면, PR을 `main` 브랜치에 병합.

### **Step 7: 뒷정리**

`main` 브랜치에 코드가 모두 합쳐졌으니, 이제 각자 로컬에 남아있는 작업용 브랜치는 삭제해줌.

```bash
# 1. 다시 main 브랜치로 돌아와서
git checkout main

# 2. 로컬에 있던 작업용 브랜치 삭제
git branch -d feature/new-model
```
-----



## ⚪ 브랜치 네이밍 컨벤션

### 기본 형식
> type/name-task

### 타입 예시
- **feat**: 기능 개발
- **fix**: 버그 수정
- **exp**: 실험
- **refactor**: 코드 구조 정리
- **doc**: 문서화
- **chore**: 빌드, CI 등 설정 관련
- **test**: 테스트 코드

### 예시
- feat/soohyun-data-preprocessing
- fix/jinseok-training-bug
- exp/gs-yolov5-sweep
- doc/soohyun-add-instructions


## ⚪ 클라우드 활용
- git에 올라가지 않는 파일들(.pt, .csv, 각종 이미지 파일)은 구글 드라이브로 공유
