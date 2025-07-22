# 3팀 프로젝트 일지
프로젝트1 기간 동안 같이 업데이트하는 프로젝트 협업 일지
## 팀원
전수현
조계승
배진석
조은영
이주성

---

---

## 2025.07.15 화
### 워크플로우 실습
    - `Collaborator 모델`을 활용한 협업 계획 및 실습
    - 프로젝트 기간 동안 레포지 pull & PR(pull Request) 계획 수립

- 첫 미팅은 7/17 (목요일) ⭐

---
- 첫 미팅은 7/17 (목요일) ⭐

---

## 2025.07.16 수
## 2025.07.16 수
### 프로젝트 계획
    (1) 프로젝트에서 활용될 Workflow(Collaborator 모델) 실습해보기
    (2) 각자 GPU현황  
        - 수현: 로컬 GTX1650
        - 계승: 맥북 프로 M4
        - 진석: RTX 4060 Ti
        - 은영: COLAB

    (3) 이번 과제의 전체적인 파이프라인 파악 및 설계 (2~3일) -> 주피터 노트북
        파이프라인 완성 -> 미팅 -> 역할 분배

    (4) 코드 모듈화(`train.py`, `model.py`, `config.py`)

    (5) 실험 및 프로젝트 마무리

자세한 사항은 https://soohyun-chris-jeon.github.io/posts/project-management/ 정리

---

## 2025.07.17 목
### 현황 파악
- 수현: 로컬 GTX1650 + COLAB pro 보류
- 계승: 세팅 보류
- 은영: COLAB
- 진석: 로컬 세팅 완료

---
자세한 사항은 https://soohyun-chris-jeon.github.io/posts/project-management/ 정리

---

## 2025.07.17 목
### 현황 파악
- 수현: 로컬 GTX1650 + COLAB pro 보류
- 계승: 세팅 보류
- 은영: COLAB
- 진석: 로컬 세팅 완료

---

### 프로젝트 역할 분배 
#### 🟣 Project Manage - (수현)
#### 🟣 Project Manage - 수현
- 프로젝트의 협업 과정을 매니징하는 역할
- 애자일/스프린트 방식으로 단위를 분리하고 회의를 주도
- 깃헙 레포지 관리, Pull request 컨펌 & merge

#### 🟣 Data Engineer - 1명 (은영)
- 깃헙 레포지 관리, Pull request 컨펌 & merge

#### 🟣 Data Engineer - 1명(은영)
- 데이터 수집, 정제 및 전처리를 담당하고 데이터 파이프라인을 구축
- 데이터 불균형 처리
- Augmentation 기법
- Data imbalance

#### 🟣 Model Architect	- 2명 (계승, 진석)
- 데이터 불균형 처리
- Augmentation 기법
- Data imbalance

#### 🟣 Model Architect	- 2명(계승, 진석)
- 딥러닝 모델을 설계하고 아키텍처를 결정
- YOLO 계열 (진석)
- R-CNN/DETR 계열 **(계승)**

#### 🟣 Experimentation Lead - (수현)
- YOLO 계열 (진석)
- R-CNN/DETR 계열 **(계승)**

#### 🟣 Experimentation Lead - 1명
- 다양한 실험을 주도하고, 하이퍼파라미터 튜닝 및 모델 성능 평가를 담당
- 학습 방식 고도화
- 평가 매트릭 
- 결과 시각화

---
### Future work
- baseline 코드 완성 ⭐⭐⭐
- 멘토링 7/20(일)
- 7/21 월

- 학습 방식 고도화
- 평가 매트릭 
- 결과 시각화


### Future work
- baseline 코드 완성 ⭐⭐⭐
- 멘토링 7/20(일)
- 7/21 월

---



## 2025.07.21 월
### 현황 파악
- 수현: **Baseline 코드** 완성(주피터 노트북). 모듈화 작업 진행 필요
- 계승: 데이터 전처리
- 은영: 
- 진석: YOLO 코드 학습까지

### Baseline 코드 특징(ver.250721)
- .ipynb 파일. 아직 모듈화 작업 동기화 x
- json 어노테이션 데이터 parsing을 하여 DataFrame 형태로 받아옴
- Data Augmentation 부분에서 torchvision이 아닌 **Albumentations**를 활용함. Object Detection에서 bbox도 안정적으로 변환이 가능한 기법
- **StratifiedGroupKFold**로 데이터 스플릿. 데이터 클래스의 불균형과 데이터 누수를 고려한 버전


### 앞으로 진행방향
- Baseline 코드를 중심으로 본인의 역할을 수행
- 대부분의 실험은 주피터 노트북으로 진행하며, 유의미한 결과가 있으면 깃허브 PL

- 수현: 
    1. train.py, eval.py 고도화하는 작업
        * earlystopping, optimizer + scheduler 조합 찾는거
        * mAP 말고 다른 
        * visualization 추가    
    2. 전체 모듈화 작업, 코드(깃허브) 관리

- 계승:
    1. 10개 후보, 앙상블

- 진석:
    1. YOLO 모델에 맞춰서 parsing하는 작업이 필요