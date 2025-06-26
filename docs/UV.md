## Table of Contents
- [🚀필수](#필수)
    - [⚙️UV 설치](#️uv-installation)
    - [🕹️aistages 환경](#️aistages-서버-사용할-때--container에-직접-설치)
    - [🖥️local 환경](#️local-환경에서-실험할-때--venv-사용)
- [📦추가 정보](#추가-정보)

# 🚀필수
- UV Installation, uv sync를 통해 가상환경을 만드세요!

## ⚙️UV Installation
- 터미널의 위치를 $HOME으로 변경 : `cd ~/`
- curl 명령어 설치 : `apt install curl`
-  🖥️ Mac
`brew install uv`
-  🐧 Linux / Ubuntu / Mac / WSL
`curl -LsSf https://astral.sh/uv/install.sh | sh`
- 🪟 Windows powershell
`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- 설치가 완료되었다면, `uv --version`을 통해 uv 명령어가 잘 인식되는지 확인합니다. 만약 `uv command is not found` 에러가 발생했다면 uv 환경변수 추가를 진행합니다.

### uv 환경변수 추가
- aistages 서버라면, uv 실행파일이 `/data/ephemeral/home/.local/bin`에 있을 겁니다.
    `vim /root/.bashrc` 를 통해 환경설정 파일을 편집합니다.
- .bashrc 파일의 맨 마지막줄에 다음과 같이 추가하고, `:wq` 를 입력하여 저장합니다.
    ```bash
    export PATH="/data/ephemeral/home/.local/bin:$PATH"
    ```
- `source .bashrc` 명령어를 통해 `.bashrc` 파일을 현재 terminal 세션에 바로 적용합니다.
- `uv --version`을 통해 uv 명령어가 잘 작동하는지 확인합니다.

## 🕹️aistages 서버 사용할 때 > container에 직접 설치
- `uv pip install -r uv.lock`

## 🖥️Local 환경에서 실험할 때 > venv 사용
### 🔄Sync 의존성 동기화
- `uv sync` : `uv.lock` 파일이 존재한다면, `uv pip install` 없이 이를 통해 .venv/를 구성할 수 있다.

### 🔋Activate venv
- 🖥️ Mac
`source .venv/bin/activate`
- 🐧 Linux / Ubuntu / Mac / WSL
`source .venv/bin/activate`
- 🪟 Windows powershell
`.venv\Scripts\Activate.ps1`
- 🔙 비활성화는 모두 공통 : `deactivate`


# 📦추가 정보

## 🛠️ init & install venv
### 🧱 venv 시작
- `uv venv --python 3.11` : python 3.11 버전으로 가상환경을 만든다.
- `uv pip install -r requirements.txt` : PyTorch를 CUDA 11.8버전용으로 다운받아아 하므로 requirements.txt를 먼저 설치한다.
- `uv pip install -r pyproject.toml` : 나머지 의존성을 설치한다.

### 🔋venv 활성화 / 비활성화
- 🖥️ Mac
`source .venv/bin/activate`
- 🐧 Linux / Ubuntu / Mac / WSL
`source .venv/bin/activate`
- 🪟 Windows powershell
`.venv\Scripts\Activate.ps1`
- 비활성화는 모두 공통 : `deactivate`

## 📥 의존성 관리
### 📌lock 파일 생성
- `uv lock` : 현재 venv 환경을 `uv.lock` 파일로 생성한다.

### 🔄lock 파일로부터 venv 구성하기
- `uv sync` : `uv.lock` 파일이 존재한다면, `uv pip install` 없이 이를 통해 .venv/를 구성할 수 있다.

### ➖특정 의존성 제거
- `uv remove <package>` : 해당 의존성을 제거하고 pyproject.toml, uv.lock을 업데이트하는 명령어
    - e.g. `uv remove pandas`

### ➕특정 의존성 추가
- `uv add <pacakge>` : 해당 의존성을 추가하고 pyproject.toml, uv.lock을 업데이트하는 명령어
    - e.g. `uv add imgaug`

## 🧩 기타
### 🌳의존성 시각화
- `uv tree` : dependency tree를 터미널에 출력
### 📦빌드
- `uv build` & `uv publish` : uv 프로젝트 폴더를 압축하고, build 파일을 PyPI에 업로드한다.