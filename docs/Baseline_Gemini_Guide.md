+) 해당 가이드는 python script를 사용할 때의 가이드입니다.

## 🔻yaml 파일 설정
#### 1. config.yaml 파일 복사 후 설정 변경
a. `project/codes/config.yaml` 파일을 복사하여 `project/codes/practice` 디렉토리에 붙여넣는다.

b. `config.yaml`파일 이름을 실험 내용에 맞게 수정한다.

#### 2. ⚠️ config.yaml 수정 시 주의사항 

- `fine_tuning` : (25.07.03) 기준 'full'로 설정하는 것을 권장
    - 'head', 'custom', 'scatch' 시 모델에 따라 직접 구현하거나 잘 작동하는지 확인 필요

- `image_size` : 224, 384 중 하나를 권장. 384 이상으로 설정하면 OOM 문제 발생 가능.

- `norm_mean`, `norm_std` : 
    - `fine_tuning`을 'full' 또는 'scratch'로 설정했거나 `pretrained`를 False로 설정했다면 mean, std 둘 다 `[0.5, 0.5, 0.5]`로 하는 게 좋습니다. 
    - `fine_tuning`을 'head'로 설정했다면, backbone 모델의 pretrained norm_mean, norm_std를 알아내야 합니다.

- `class_imbalance` : 클래스 불균형한 데이터를 증강시키는 옵션입니다. 
    - 아래처럼 주석으로 세부 옵션을 제거하면 클래스 불균형 데이터를 증강시키지 않습니다.
    ```
    class_imbalance: 
        # aug_class: [1, 13, 14]
        # max_samples: 78
    ```
    - 세부 옵션을 작성하면 클래스 불균형 데이터를 offline 증강합니다. 생성된 이미지는 파일 실행이 종료되고 자동으로 삭제됩니다. 세부 옵션은 `val_split_ratio: 0.15`에 맞게 설정되었으므로 수정하면 오류가 발생할 수 있습니다.
    ```
    class_imbalance: 
        aug_class: [1, 13, 14]
        max_samples: 78
    ```

- `timm` : timm backbone 모델 로딩 시 설정하는 옵션입니다.
    - `activation` : `timm.create_model`에 전달되는 옵션으로, backbone 모델의 activation layer를 변경할 수 있습니다. 그러나 종종 "GELU"로 설정 시 오류가 발생합니다. "None"을 입력하면 기본값을 사용합니다.
    - `head` : timm model 중 Classifier Head가 있는 것, 없는 것이 있습니다. 모델 구조를 먼저 파악한 후 이 옵션을 사용하세요.

- `custom_layer` : 세부 옵션 설정 시 `TimmWrapper`를 사용해 Classifier Head를 커스터마이징 합니다.

- `batch_size` : 64를 초과하여 설정할 시, `image_size`와 모델 파라미터 규모에 따라 OOM 문제가 발생할 수 있습니다.

- `patience` : 30~50 값을 권장합니다.

- `wandb`
    - `log` : False로 설정할 시, wandb를 사용하지 않습니다. > 연습할 때 False로 설정하세요.


## 🔻gemini_main.py 실행
- ⚠️ 주의) gemini_main.py 실행 시, `*.yaml` 설정 파일은 항상 `project/codes`  폴더보다 아래에 존재해야 합니다.
    ```
    project/codes/config.yaml [o]
    project/codes/practice/config.yaml [o]
    project/codes/song/config.yaml [o]

    project/config.yaml [x]
    project/data/config.yaml [x]
    ```
1. 터미널에서 `gemini_main.py` 파일 위치로 이동합니다.
    ```bash
    cd codes/
    ```

2. `gemini_main.py` 파일을 백그라운드에서 실행하면서 log 파일을 설정합니다. 이때 설정 파일의 위치를 같이 알려줍니다.
    - 현재 위치인 `projects/codes`를 기준으로 상대위치를 전달합니다.
    ```bash
    # e.g. config.yaml 파일의 위치가 project/codes/song/config_resnet50.yaml 이라고 가정.
    nohup python gemini_main.py --config song/config_resnet50.yaml > ../logs/2507032355_resnet50_384_full_adamW_Cosine_offaug-eda-dilation-erosion_batch64.log 2>&1 &
    ```
    - 또는 그냥 실행한다.
    ```bach
    python gemini_main.py --config song/config_resnet50.yaml
    ```

3. [팀 Notion](https://www.notion.so/22540cb3731d800a9b19d62dad7d7f43?v=22540cb3731d8088a0c3000c572f66f5&source=copy_link)에 실험 정리.

