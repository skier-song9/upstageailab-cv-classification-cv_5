# 🤝 5조 팀 프로젝트 Git 협업 가이드

## 📋 개요

이 문서는 **Upstage AI Lab CV 분류 대회 5조**의 Git 협업 워크플로우를 상세히 설명합니다. 각 조원이 개인 계정에서 작업하고 팀 계정으로 기여하는 전체 과정을 담고 있습니다.

### 🎯 협업 구조
```
팀 계정 (upstream) ← PR/Merge ← 개인 계정 (origin) ← 로컬 작업
AIBootcamp13/upstageailab-cv-classification-cv_5    your-username/upstageailab-cv-classification-cv_5    로컬 저장소
```

---

## 🚀 1단계: 초기 설정

### 1.1 팀 저장소 Fork

1. **GitHub에서 팀 저장소 접속**
   ```
   https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5
   ```

2. **Fork 버튼 클릭**
   - 우측 상단의 "Fork" 버튼 클릭
   - 본인의 GitHub 계정으로 Fork
   - 결과: `your-username/upstageailab-cv-classification-cv_5` 생성

### 1.2 로컬에 클론

```bash
# 본인의 Fork된 저장소를 로컬에 클론
git clone https://github.com/your-username/upstageailab-cv-classification-cv_5.git

# 디렉토리 이동
cd upstageailab-cv-classification-cv_5
```

### 1.3 Remote 설정

```bash
# 현재 remote 확인
git remote -v

# upstream(팀 계정) 추가
git remote add upstream https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5.git

# 설정 확인
git remote -v
```

**올바른 설정 결과:**
```
origin    https://github.com/your-username/upstageailab-cv-classification-cv_5.git (fetch)
origin    https://github.com/your-username/upstageailab-cv-classification-cv_5.git (push)
upstream  https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5.git (fetch)
upstream  https://github.com/AIBootcamp13/upstageailab-cv-classification-cv_5.git (push)
```

---

## 🌟 2단계: 브랜치 전략

### 2.1 브랜치 명명 규칙

```bash
# 기능별 브랜치 명명 예시
feature/jayden-data-augmentation     # 데이터 증강 작업
feature/jayden-model-optimization    # 모델 최적화
feature/jayden-visualization        # 시각화 작업
bugfix/jayden-training-loop         # 버그 수정
experiment/jayden-new-architecture  # 실험적 기능
```

### 2.2 브랜치 생성 및 전환

```bash
# upstream에서 최신 변경사항 가져오기
git fetch upstream
git checkout main
git merge upstream/main

# 새 브랜치 생성 및 전환
git checkout -b feature/jayden-data-augmentation

# 브랜치 확인
git branch
```

---

## 💻 3단계: 개발 워크플로우

### 3.1 일반적인 작업 흐름

```bash
# 1. upstream에서 최신 변경사항 동기화
git fetch upstream
git checkout main
git merge upstream/main

# 2. 작업 브랜치로 전환 (또는 새로 생성)
git checkout feature/jayden-data-augmentation
# 또는
git checkout -b feature/jayden-data-augmentation

# 3. main의 최신 변경사항을 작업 브랜치에 반영
git merge main

# 4. 작업 수행 (코드 작성, 파일 수정 등)
# ... 코딩 작업 ...

# 5. 변경사항 확인
git status
git diff

# 6. 변경사항 스테이징
git add .
# 또는 특정 파일만
git add codes/baseline_improved.py
git add docs/experiment-results.md

# 7. 커밋
git commit -m "feat: 데이터 증강 기능 추가

- Albumentations 라이브러리를 사용한 다양한 증강 기법 구현
- 문서 이미지에 적합한 증강 파라미터 설정
- 훈련/검증 transform 분리
- 성능 향상: 75% → 82% accuracy"

# 8. 원격 저장소에 푸시
git push origin feature/jayden-data-augmentation
```

### 3.2 커밋 메시지 컨벤션

```bash
# 커밋 타입 예시
feat: 새로운 기능 추가
fix: 버그 수정
docs: 문서 수정
style: 코드 포맷팅, 세미콜론 누락 등
refactor: 코드 리팩토링
test: 테스트 코드 추가
chore: 빌드 업무 수정, 패키지 매니저 수정

# 좋은 커밋 메시지 예시
git commit -m "feat: ResNet34 모델에 validation 기능 추가

- train/validation split 구현 (80:20 비율)
- validation accuracy 모니터링 추가
- early stopping 기능 구현
- 최고 성능 모델 자동 저장

Closes #15"
```

---

## 🔄 4단계: Pull Request 생성

### 4.1 GitHub에서 PR 생성

1. **GitHub에서 본인의 Fork 저장소 접속**
   ```
   https://github.com/your-username/upstageailab-cv-classification-cv_5
   ```

2. **"Compare & pull request" 버튼 클릭**
   - 푸시 후 자동으로 나타나는 버튼 클릭
   - 또는 "Pull requests" 탭에서 "New pull request" 클릭

3. **PR 정보 작성**
   ```markdown
   ## 📋 작업 내용
   데이터 증강 기능을 추가하여 모델 성능을 개선했습니다.

   ## 🔧 주요 변경사항
   - [ ] Albumentations 라이브러리를 사용한 데이터 증강 구현
   - [ ] 문서 이미지에 적합한 증강 파라미터 설정
   - [ ] Train/Validation transform 분리
   - [ ] 증강 효과 검증 및 성능 측정

   ## 📊 성능 결과
   - 베이스라인: 75% accuracy
   - 개선 후: 82% accuracy (+7% 향상)

   ## 🧪 테스트
   - [ ] 로컬에서 학습 파이프라인 테스트 완료
   - [ ] 다양한 증강 조합 실험 완료
   - [ ] 과적합 여부 확인

   ## 📸 스크린샷 / 결과물
   (필요시 이미지나 그래프 첨부)

   ## 🔗 관련 이슈
   Closes #15

   ## 🏷️ 라벨
   - Type: Feature
   - Priority: High
   - Reviewer: @team-leader-username
   ```

### 4.2 PR 체크리스트

```markdown
## ✅ PR 전 체크리스트

### 코드 품질
- [ ] 코드가 정상적으로 실행되는가?
- [ ] 주석이 적절히 작성되었는가?
- [ ] 변수명과 함수명이 명확한가?
- [ ] 불필요한 코드가 제거되었는가?

### 기능 검증
- [ ] 새로운 기능이 의도대로 작동하는가?
- [ ] 기존 기능에 영향을 주지 않는가?
- [ ] 에러 핸들링이 적절한가?

### 문서화
- [ ] README.md 업데이트 (필요시)
- [ ] 새로운 기능에 대한 문서 작성
- [ ] 코드 주석 추가

### Git
- [ ] 커밋 메시지가 명확한가?
- [ ] 불필요한 파일이 포함되지 않았는가?
- [ ] .gitignore 확인
```

---

## 🔀 5단계: 코드 리뷰 및 Merge

### 5.1 리뷰 요청

```bash
# 팀원들에게 리뷰 요청
# GitHub에서 Reviewers 지정
# Slack이나 Discord로 리뷰 요청 알림
```

### 5.2 리뷰 반영

```bash
# 리뷰 피드백 반영 후 추가 커밋
git add .
git commit -m "fix: 리뷰 피드백 반영

- 변수명 명확하게 수정
- 에러 핸들링 추가
- 코드 주석 보강"

git push origin feature/jayden-data-augmentation
```

### 5.3 Merge 후 정리

```bash
# Merge 후 로컬 정리
git checkout main
git fetch upstream
git merge upstream/main

# 작업 완료된 브랜치 삭제
git branch -d feature/jayden-data-augmentation

# 원격 브랜치도 삭제 (선택사항)
git push origin --delete feature/jayden-data-augmentation
```

---

## 🔄 6단계: 정기적인 동기화

### 6.1 매일 작업 시작 전

```bash
# 1. main 브랜치로 전환
git checkout main

# 2. upstream에서 최신 변경사항 가져오기
git fetch upstream

# 3. 로컬 main을 upstream main과 동기화
git merge upstream/main

# 4. 본인의 origin main도 업데이트
git push origin main

# 5. 작업 브랜치에도 최신 변경사항 반영
git checkout feature/jayden-current-work
git merge main
```

### 6.2 충돌 해결

```bash
# 충돌 발생 시
git merge main
# Auto-merging conflict가 발생하면

# 1. 충돌 파일 확인
git status

# 2. 충돌 파일 수동 편집
# <<<<<<< HEAD
# 현재 브랜치의 내용
# =======
# 병합하려는 브랜치의 내용
# >>>>>>> main

# 3. 충돌 해결 후 커밋
git add .
git commit -m "resolve: merge conflict with main branch"
```

---

## 📁 7단계: 프로젝트 구조 관리

### 7.1 권장 폴더 구조

```
upstageailab-cv-classification-cv_5/
├── codes/
│   ├── baseline/
│   │   └── original_baseline.ipynb
│   ├── experiments/
│   │   ├── jayden_data_augmentation.ipynb
│   │   ├── member2_model_optimization.ipynb
│   │   └── member3_ensemble.ipynb
│   └── final/
│       └── best_solution.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── submissions/
├── docs/
│   ├── git-workflow/
│   ├── experiments/
│   └── final-report/
├── models/
│   ├── checkpoints/
│   └── final/
└── results/
    ├── experiments/
    └── final/
```

### 7.2 파일 명명 규칙

```bash
# 코드 파일
jayden_baseline_v1.ipynb
jayden_data_augmentation_v2.py
team_ensemble_final.py

# 모델 파일
resnet34_jayden_epoch10_acc82.pth
efficientnet_b0_team_best.pth

# 결과 파일
submission_jayden_20250701.csv
experiment_results_data_aug.md
```

---

## 🚨 8단계: 주의사항 및 베스트 프랙티스

### 8.1 절대 하지 말아야 할 것

```bash
# ❌ 절대 금지
git push --force origin main  # 강제 푸시 금지
git commit -am "quick fix"    # 의미없는 커밋 메시지
git add . && git commit -m "." # 모든 파일 무분별 커밋

# ❌ 직접 main 브랜치에 푸시 금지
git checkout main
git commit -m "direct commit to main"  # 위험!
git push origin main
```

### 8.2 권장사항

```bash
# ✅ 권장
# 작은 단위로 자주 커밋
git add specific_file.py
git commit -m "feat: specific feature implementation"

# 의미있는 브랜치명 사용
git checkout -b feature/jayden-validation-implementation

# 정기적인 동기화
git fetch upstream && git merge upstream/main
```

### 8.3 데이터 파일 관리

```bash
# .gitignore에 추가해야 할 항목들
*.pth          # 모델 체크포인트
*.pkl          # 피클 파일
data/train/    # 대용량 데이터
data/test/
__pycache__/   # Python 캐시
.DS_Store      # macOS 시스템 파일
*.log          # 로그 파일

# Git LFS 사용 (대용량 파일)
git lfs track "*.pth"
git lfs track "*.pkl"
git add .gitattributes
```

---

## 🛠️ 9단계: 실무 시나리오별 가이드

### 9.1 새로운 실험 시작

```bash
# 시나리오: 새로운 모델 아키텍처 실험
git checkout main
git pull upstream main
git checkout -b experiment/jayden-efficientnet-b4

# 작업 후
git add experiments/efficientnet_b4_experiment.ipynb
git commit -m "experiment: EfficientNet-B4 아키텍처 테스트

- timm 라이브러리를 사용한 EfficientNet-B4 모델 구현
- 베이스라인 대비 파라미터 수 증가: 19M
- 초기 실험 결과: validation accuracy 78%
- 추가 튜닝 필요"

git push origin experiment/jayden-efficientnet-b4
```

### 9.2 버그 수정

```bash
# 시나리오: 데이터 로더에서 메모리 누수 발견
git checkout main
git pull upstream main
git checkout -b bugfix/jayden-dataloader-memory-leak

# 수정 후
git add codes/dataset.py
git commit -m "fix: 데이터로더 메모리 누수 문제 해결

- ImageDataset.__getitem__에서 이미지 파일 핸들 누수 수정
- PIL Image 객체 명시적 close() 추가
- 메모리 사용량 50% 감소 확인
- 장시간 학습 시 안정성 향상

Fixes #23"

git push origin bugfix/jayden-dataloader-memory-leak
```

### 9.3 문서 업데이트

```bash
# 시나리오: 실험 결과 문서화
git checkout -b docs/jayden-experiment-results

git add docs/experiments/data-augmentation-results.md
git commit -m "docs: 데이터 증강 실험 결과 문서화

- 5가지 증강 기법별 성능 비교 결과 추가
- 최적 증강 조합 가이드라인 작성
- 시각화 그래프 및 예시 이미지 포함
- 팀원들을 위한 사용법 가이드 추가"

git push origin docs/jayden-experiment-results
```

### 9.4 긴급 핫픽스

```bash
# 시나리오: 제출 직전 크리티컬 버그 발견
git checkout main
git pull upstream main
git checkout -b hotfix/submission-format-error

# 빠른 수정
git add codes/inference.py
git commit -m "hotfix: 제출 파일 형식 오류 수정

- CSV 헤더 대소문자 오류 수정: 'ID' -> 'id'
- 제출 형식을 대회 요구사항에 맞게 조정
- 긴급 수정으로 테스트 제한적"

git push origin hotfix/submission-format-error

# 즉시 PR 생성하여 빠른 리뷰 요청
```

---

## 📊 10단계: 성과 추적 및 관리

### 10.1 실험 로그 관리

```bash
# experiments 폴더에 실험별 문서 작성
docs/experiments/
├── 2025-07-01-jayden-baseline-analysis.md
├── 2025-07-02-jayden-data-augmentation.md
├── 2025-07-03-jayden-model-optimization.md
└── 2025-07-04-team-ensemble-results.md
```

### 10.2 팀 성과 대시보드

```markdown
# Team Performance Tracker

## 📈 Current Best Results
- **Highest Validation Accuracy**: 85.2% (jayden-ensemble-v3)
- **Best Public LB Score**: 84.7% (team-final-submission-v2)
- **Most Stable Model**: ResNet34 + Data Aug (83.1% ± 1.2%)

## 🏆 Individual Contributions
| Member | Best Contribution | Score | Date |
|--------|------------------|-------|------|
| Jayden | Data Augmentation | +7% | 2025-07-02 |
| Member2 | Model Optimization | +5% | 2025-07-03 |
| Member3 | Ensemble Method | +3% | 2025-07-04 |

## 🎯 Next Steps
- [ ] TTA (Test Time Augmentation) 구현
- [ ] Cross-validation 정확도 검증
- [ ] 최종 앙상블 모델 구성
```

---

## 🎯 11단계: 대회 종료 전 체크리스트

### 11.1 최종 제출 준비

```bash
# 1. 최종 모델 및 코드 정리
git checkout -b final/team-submission

# 2. 최종 모델 파일 정리
mkdir models/final/
cp best_models/* models/final/

# 3. 재현 가능한 코드 작성
python codes/final/reproduce_results.py

# 4. 최종 문서 작성
git add docs/final-report/
git commit -m "docs: 최종 보고서 및 재현 코드 완성"

# 5. 최종 제출
git push origin final/team-submission
```

### 11.2 코드 정리 체크리스트

- [ ] 모든 노트북이 정상 실행되는가?
- [ ] 불필요한 실험 코드 제거
- [ ] 주석 및 문서화 완료
- [ ] 재현 가능한 스크립트 제공
- [ ] 모델 파일 정리 및 백업
- [ ] 최종 성능 결과 문서화

---

## 🆘 문제 해결 FAQ

### Q1: "git push"가 거부당할 때
```bash
# 에러: Updates were rejected because the remote contains work
git fetch origin
git merge origin/your-branch-name
# 충돌 해결 후
git push origin your-branch-name
```

### Q2: 실수로 main에 직접 커밋했을 때
```bash
# 마지막 커밋을 되돌리기 (아직 push 안한 경우)
git reset --soft HEAD~1

# 새 브랜치 생성하여 이동
git checkout -b feature/jayden-accidental-commit
git push origin feature/jayden-accidental-commit
```

### Q3: 대용량 파일을 실수로 커밋했을 때
```bash
# Git LFS로 전환
git lfs track "*.pth"
git add .gitattributes
git add large_file.pth
git commit -m "feat: migrate large files to Git LFS"
```

### Q4: upstream과 많이 뒤쳐졌을 때
```bash
# 강제 동기화 (주의: 로컬 변경사항 손실 가능)
git fetch upstream
git checkout main
git reset --hard upstream/main
git push origin main --force-with-lease
```

---

## 🎉 마무리

이 가이드를 따라 하시면 팀 프로젝트에서 효율적이고 안전한 Git 협업이 가능합니다. 

### 📞 도움이 필요할 때
1. **Git 명령어 헬프**: `git --help <command>`
2. **팀 슬랙/디스코드**: 언제든 질문하세요!
3. **GitHub 이슈**: 기술적 문제는 이슈로 등록

### 🏆 성공적인 협업을 위한 마지막 팁
- **소통이 핵심**: 작업 전 팀원들과 상의
- **작은 단위로 자주**: 큰 변경사항보다 작은 PR이 좋음
- **문서화 습관**: 미래의 나와 팀원을 위해
- **실험 정신**: 새로운 시도를 두려워하지 마세요!

**5조 화이팅! 🚀**

---

*📅 작성일: 2025년 7월 1일*  
*✍️ 작성자: AI Assistant*  
*📁 파일 위치: `/docs/git-workflow/team-collaboration-guide.md`*
*🔄 최종 수정: 대회 진행에 맞춰 지속 업데이트*
