# KoBertSum: BertSum 기반의 한국어 요약 모델

[AI Hub 문서요약 텍스트](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=97)에서 학습시킨 BERT 기반 요약 모델, [BertSum](https://arxiv.org/pdf/1908.08345.pdf)의 공개 코드입니다. 현재 공개된 모델은 BertSum-Ext이며, 생성 요약 모델은 추후 공개할 예정입니다.

요약 분야의 많은 연구 및 구현들이 여전히 [nlpyang/PreSumm](https://github.com/nlpyang/PreSumm) 코드를 사용합니다. 하지만 해당 코드는 매우 낮은 단계에서부터 구현되었습니다. 본 구현은 추출 요약의 재생산성을 높이기 위해 직접 작성한 코드입니다.

AI Hub 문서요약 텍스트 데이터셋은 신문기사, 기고문, 잡지, 법원 판결문으로부터 발췌한 400k개 텍스트로 구성된 데이터셋입니다. CNN / DailyMail 등의 영문 벤치마크 데이터셋과 다르게, 본 데이터셋은 생성 요약 레이블과 추출 요약 레이블을 모두 가지고 있습니다. 생성 요약 레이블은 1개 문장으로 작성되었으며, 추출 요약 레이블은 원문 내용 및 문장 순서를 고려한 3개의 문장 인덱스로 구성되어 있습니다.  

## Training
한국어 기반의 BERT는 [klue/bert-base](https://huggingface.co/klue/bert-base)로부터 가져왔으며, 학습은 T4 1대에서 진행되었습니다.  

(추후 wandb 그래프 첨부)

추출 요약 레이블이 주어지지 않은 대부분의 영문 벤치마크 데이터셋(eg. CNN / DM, XSum, WikiHow, etc.)과 다르게, 본 데이터셋은 사람이 직접 작성한 추출 요약 레이블을 가지고 있습니다. 이로 인해 모델 학습 시, 추출 요약 모델을 binary classification으로 학습시키기 위한 별도의 오라클 알고리즘이 필요하지 않습니다.  


## Evaluation
AI Hub 문서요약 텍스트의 검증 데이터셋에서 평가한 KoBertSum의 성능은 다음과 같습니다.

|rouge1|rouge2|rougeL| 
|:---:|:---:|:---:|
|72.25|64.21|60.78|

ROUGE 스코어는 [google-research](https://github.com/google-research/google-research/blob/master/rouge/rouge_scorer.py)에서 제공하는 라이브러리, [rouge-score](https://pypi.org/project/rouge-score/)로 계산되었습니다. 논문에 비해 ROUGE 스코어가 높은 이유는, 영어 기반 요약 모델의 경우 생성 요약 레이블에 대해 평가되는 반면, KoBertSum은 이미 주어진 추출 요약 레이블에 대해 평가되기 때문입니다.  

rouge-score은 ROUGE-L을 계산을 위해 rougeL, rougeLsum 2개의 스코어를 제공합니다. 두 스코어의 차이는 다음과 같습니다.  
* rougeL : 문자열 내 문장 구분자 `\n`을 무시  
* rougeLsum : 문자열 내 문장 구분자 `\n`을 고려  

[nlpyang/PreSumm](https://github.com/nlpyang/PreSumm)이 사용한 라이브러리 pyrouge에서 제공하는 ROUGE-L 스코어는 rouge-score의 rougeLsum과 대응됩니다. 본 코드에서는 rougeL만을 이용해 ROUGE-L 스코어를 계산했습니다.  


## Usage
직접 모델을 학습시키고 싶다면 실험을 yaml 파일로 정의한 뒤, 아래 명령어를 실행합니다.
```
python train.py -—config-name exp_0
```

모델을 평가하고 싶다면, 실험 파일의 `test_checkpoint`에 체크포인트 경로를 입력한 뒤, 아래 명령어를 실행합니다.
```
python test.py —-config-name exp_0
```

## License
BSD 3-Clause License Copyright (c) 2022
