### Code Structure

```
${AutoML}
├── config/
│   └── sample.yaml
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   └── param_config/
│       └── param_config.py
├── output/
│   └── result.csv
├── preprocessing.py
├── README.md
├── run.py
└── utils.py
```
- config: 모델 관리를 위한 파라미터를 설정하는 YAML 파일 경로
- data: 데이터 경로
- models: 모델 저장 및 모델 파라미터 설정 경로
      - param_config.py: 튜닝을 위한 하이퍼 파라미터 범위가 지정된 파일
- output: 실행 결과가 저장되는 경로
- preprocessing.py: 데이터 전처리 코드
- run.py: 실제 실행 코드
- utils.py: run.py를 실행하기 위한 다양한 기능들이 포함된 코드

### run

1. 'data/'에 사용할 데이터 업로드 ex) train.csv, test.csv
2. 'config/sample.yaml'을 복사하여 업로드한 데이터 맞게 수정
3. predict.py 실행
```
python run.py --config sample.yaml
```

### 실험 관리

1. YAML 파일 하나가 1개의 실험에 대응됨
2. YAML 파일의 data 부분에 사용할 데이터셋의 파일명 기입
3. YAML 내부에서 적용할 모델, metric, 기법, seed 등을 설정
4. output 파일에 해당 YAML 파일에 대응되는 실험결과 생성 (생성될 당시의 시간 자동기입됨)
