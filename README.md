# Contextualized and Aligned Audio-Text Fusion Models for Emotion Recognition

[제2회 ETRI 휴먼이해 인공지능 논문경진대회](#https://aifactory.space/competition/detail/2234)

대화 속에서의 감정을 예측하기 위해 음성 데이터와 텍스트 데이터를 함께 활용한 **CASE: Contextualized and Aligned Speech Embedding** 을 제시한다.


<img width="1039" alt="image" src="https://user-images.githubusercontent.com/53552847/230810083-4f9878dc-e85d-4666-a5dd-866c87b5a7c2.png">

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


### Run

```python
# In addition to this, there are various arguments, so please refer to the parser at "main.py"
python3 main.py --model {model_name} --wandb_project {your_project_name} --wandb_entity {your_entity_name} --wandb_name {saved_wandb_model_name}
```
##### pet_train 돌리기
```sh
chmod +x pet_train
./pet_train.sh
```

### 순서
```
1. make_data.py ( annotation -> data.csv 생성)
2. split.py (train-test를 Session 기준, Label별 비율을 유지하도록 32:8로 split)
3. train.sh 실행
4. test.sh 실행
```