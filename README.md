### Code Structure

```
${AutoBaseline}
├── config/
│   └── baseline.yaml
├── data/
│   └── data.csv
├── output/
│   └── result.csv
├── README.md
├── run.py
└── utils.py
```
- config: 모델 관리를 위한 파라미터를 설정하는 yaml 파일 경로
- data: 데이터 경로
- output: 실행 결과가 저장되는 경로
- run.py: 실행 코드

### Run

1. 'data/'에 사용할 데이터 업로드 ex) my_data.csv
2. 'config/baseline.yaml'을 복사하여 my_data.csv에 맞게 수정 ex) my_baseline.yaml
3. 'run.py'에서 setting_name을 'my_baseline'으로 수정
4. run.py 실행
```
python run.py
```

### 실험 관리

1. yaml 파일 하나가 1개의 실험에 대응됨
2. yaml 파일의 data 부분에 사용할 데이터셋의 파일명 기입
3. 사용할 모델, metric, target, 컬럼 설정하고 seed를 설정하여 실험 환경 구축
4. output 파일에 해당 yaml 파일에 대응되는 실험결과 생성 (생성될 당시의 시간 자동기입됨)
