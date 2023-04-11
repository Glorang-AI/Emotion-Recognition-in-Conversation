# Contextualized and Aligned Audio-Text Fusion Models for Emotion Recognition

[제2회 ETRI 휴먼이해 인공지능 논문경진대회](https://aifactory.space/competition/detail/2234)

대화 속에서의 감정을 예측하기 위해 음성 데이터와 텍스트 데이터를 함께 활용한 Audio-Text Fusion Model인 **CASE: Contextualized and Aligned Speech Embedding** 을 제시한다.


<img width="1051" alt="image" src="https://user-images.githubusercontent.com/53552847/230815991-5e0c3b72-c74d-4623-ac4e-e918e6e7f81a.png">

## Setup
### Dependencies
```
pandas == 1.5.3
numpy == 1.22.0
torch == 1.13.1+cu117
transformers == 4.26.1
torcheval == 0.0.6
sklearn == 1.2.1
```

### Install Requirements
```python
bash requirements.sh
```

## Code
```
+- main.py
+- trainer.py       
+- models.py        
+- dataset.py       
+- requirements.sh  
+- train.sh         
+- test.sh          
+- utils
|   +- audio_embedding.py   # 사전학습모델을 활용하여 Wav를 저장
|   +- contrastive.py       # Contrastive Set 생성
|   +- loss.py              # Loss Function 
|   +- seed.py              # Seed Setting

+- src
|   +- make_data.py         # annotation data로 부터 `data.csv` 생성
|   +- split.py             # `data.csv`를 활용하여 Train-Test split 수행한 후 `train.csv`, `test.csv` 생성
|   +- param_count.py       # 모델별 Parameter 수 계산
|   +- all_neutral.py       # 모두 Neutral로 예측했을 때, Metric 계산

+- data
|   +- annotation
|   +- EDA
|   +- IBI
|   +- TEMP
|   +- wav
|   +- data.csv             # `src/make_data.py`로 부터 생성 - annotation 활용
|   +- train.csv            # `src/split.py`로 부터 생성 ( Session 기준 8:2로 분할 )
|   +- test.csv             # `src/split.py`로 부터 생성 ( Session 기준 8:2로 분할 )
|   +- emb_train.py         # `main.py` 실행 시 생성 ( args.mode == "train" ) - wav 활용
|   +- emb_test.py          # `main.py` 실행 시 생성 ( args.mode == "test" ) - wav 활용

+- save             # 모델 저장
+- wandb            # wandb 관련 config 저장
```

## Run

### Dataset 생성
```python
# KEMDy20 내 폴더들을 data 폴더의 하위 폴더로 이동 (Code 참고)

python3 src/make_data.py    # `data/data.csv` 생성
python3 src/split.py        # `data/train.csv`, `data/test.csv` 생성
```

### 학습 및 예측
In addition to these arguements, there are various arguments. So please refer to the parser at "main.py"
```python
# train
python3 main.py --model {model_name} --wandb_project {your_project_name} --wandb_entity {your_entity_name} --wandb_name {saved_wandb_model_name}
    # or
chmod +x train.sh
./train.sh

# test
python3 main.py --model {model_name} --mode "test" --test_model_path {save_model_path}
    # or
chmod +x test.sh
./test.sh
```

## Experiments
- Epoch 150으로 진행한 후, 매 30 epoch 마다 저장 후 test ( 최종 120 Epoch 결과 )
- `CASE (compressing)`가 Macro-F1, Micro-F1 모두에서 가장 좋은 성능을 보임.

|                           | Macro-F1 | Micro-F1 | Weighted-F1 |
| ------------------------- | -------- | -------- | ----------- |
| CASE (compressing)        | 32.82    | 43.77    | 87.89       |
| CASE (attention)          | 27.91    | 39.15    | 86.60       |
| Concat                    | 30.16    | 40.54    | 87.94       |
| MMM                       | 30.01    | 40.07    | 86.94       |
| CASE-concat (compressing) | 31.62    | 45.34    | 88.40       |
| CASE-concat (attention)   | 26.49    | 34.53    | 86.51       |

