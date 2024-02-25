### Code Structure

```
${AutoBaseline}
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
    - prediction: train, test 데이터를 활용한 실제 추론을 위한 실험 관리 저장소
    - validation: train 데이터를 모델 검증을 위한 실험 관리 저장소
- data: 데이터 경로
- output: 실행 결과가 저장되는 경로
- predict.py: 실제 추론 코드
- validate.py: 모델 검증 코드

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
