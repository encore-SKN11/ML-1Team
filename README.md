# 2ST

# ML-1Team
# EDA-1Team

# 팀명 : TNT 팀

## 팀원 소개 

| 김장수 | 이채은 | 황준호|
| --- | --- | --- |
| ![Image](https://github.com/user-attachments/assets/9a5c53e6-e3da-4810-b96a-9b376100b926) | ![Image](https://github.com/user-attachments/assets/9a5c53e6-e3da-4810-b96a-9b376100b926) | ![Image](https://github.com/user-attachments/assets/b19ee646-41db-48f0-b502-3f0c0e0ef549) |
| <div align="center">INTP</div> | <div align="center">INTP</div> | <div align="center">ENTJ</div> |
# 프로젝트 주제 : 미국 간호사 이탈 예측 프로젝트

# 📅 개발 기간
**2025.03.14 ~ 2025.03.18 (총 5일)**

##  🎯 프로젝트 목표

### **프로젝트 개요**
- 간호사의 **퇴사 여부(Attrition)** 를 예측하는 머신러닝 모델을 개발하여, 의료기관에서 인력 관리를 효과적으로 수행할 수 있도록 돕는 것이 목표이다.


### **기대효과**:

 **1. 퇴사 예측 기반 인력 관리 최적화**
- **조기 위험 인식**: 퇴사 가능성이 높은 간호사를 조기에 파악하여 선제적 대응이 가능
- **채용 전략 개선**: 향후 고용 시 이탈 가능성이 낮은 지원자를 선별하여 안정적인 인력 확보

 **2. 근무 환경 및 정책 개선**
- **효과적인 복지 정책 수립**: 근무 환경 개선을 통해 직원 만족도를 높이고, 이탈률 감소에 기여

 **3. 의료 기관 운영 효율성 증대 및 비용 절감**
- **운영 효율성 향상**: 인력 부족 문제를 방지함으로써 환자 케어의 질을 유지
- **비용 절감**: 인력 채용과 교육에 드는 비용을 절감하여 전반적인 운영 비용 최적화

### **접근 방식**:
  - 데이터를 목적에 맞게 전처리를 진행하고 여러 모델 중 학습 후 성능이 좋은 모델을 바탕으로, 퇴사 여부에 큰 영향을 미치는 feature들을 수치나 시각화(그래프 등)를 통해 확인함. 이를 통해 병원에 제공할 수 있는 인사이트를 도출
   
### **타겟 변수**: **`퇴사여부`** ('Yes': 퇴사 o , 'No': 퇴사 x)


## 📂데이터 구성 
### - 데이터소스: [Employee Attrition for Healthcare](https://www.kaggle.com/datasets/jpmiller/employee-attrition-for-healthcare/data)
- **1676명**의 데이터
- **35개의 컬럼** 중 주요 컬럼:
    - `초과근무여부`: 초과근무 여부
    - `결혼상태`: 결혼 상태
    - `나이(Age)`: 간호사의 나이
    - `마지막승진이후연수`: 마지막 승진 이후 연수
    - `현재직무근속연수`: 현재 직무에서 근속한 연수
    - `워라밸지수`: 일과 삶의 균형 지수
    - `출장빈도(BusinessTravel)`: 출장 빈도 (Rarely, Frequently 등)
    - `근무환경만족도`: 근무 환경 만족도
    - `직무만족도(JobSatisfaction)`: 직무 만족도 (1~4점 척도)
    - `출퇴근거리`: 출퇴근 거리
    - `직무몰입도`: 직무 몰입도
    - `부서`: 근무 부서
    - `현매니저와근속연수`: 현 매니저와의 근속 연수

## 🧹 EDA

### 1. 데이터 로드

![Image](https://github.com/user-attachments/assets/5dc747d4-ef28-49ac-9285-c1161f53e9a1)
 


### 2. 데이터 구조 및 기초 통계 확인
![Image](https://github.com/user-attachments/assets/b3741e48-a594-40c1-ad93-a34da12b6747)

![Image](https://github.com/user-attachments/assets/c5b12915-c720-4f0c-b8f3-fae9e3ab4634)


  
### 3. 결측치 및 이상치 탐색
![Image](https://github.com/user-attachments/assets/35c77b40-17f7-46af-b8e9-4d2d269a8910) 
  - 결측치 : 모든 컬럼에서 **결측치 없음**

![Image](https://github.com/user-attachments/assets/c038dbb8-f705-43ce-ad1c-0420f88ac0d4)
  - 이상치 : 나이,월급,현 매니저와 근속연수 에서 이상치 발견 <br/>
=> '나이' feature의 이상치 가능성 높음 : ('월급이 적다', '직무에 만족하지 못한다' 등 피처들로 인한 요인보다 나이로 인해 은퇴하는 것으로 추정)



### 4. 데이터 시각화를 통한 탐색

   #### 1️⃣퇴사 여부 분포 : 퇴사한 인원('Yes')과 퇴사하지 않은 인원('No')의 데이터 분포를 나타낸 그래프
   
  ![Image](https://github.com/user-attachments/assets/e16e2910-2214-414d-8652-0305ab6340ef)
  - 퇴사한 인원(Yes)와 퇴사하지 않은 인원(No)의 데이터 분포를 나타낸 그래프
  - 퇴사하지 않은 인원(0)의 수가 퇴사한 인원(1)보다 훨씬 많음 → **데이터 불균형 존재**


   #### 2️⃣ 수치형 변수 간 상관관계 : 지표 간의 상관관계를 나타낸 히트맵(Heatmap)
![Image](https://github.com/user-attachments/assets/d17b4262-a638-4517-99d6-2095e4073723)
  - 주요 지표들이 서로 얼마나 강한 관계를 가지는지 시각적으로 표현
  - 색상(파란색~빨간색)으로 상관계수의 크기를 나타냄
      - 빨간색(1에 가까움): 강한 양의 상관관계 (서로 비례)
      - 파란색(-1에 가까움): 강한 음의 상관관계 (서로 반비례)
      - 흰색(0에 가까움): 거의 상관없음<br/>
    
   

### 5. 데이터 정제 및 전처리
  #### 1️⃣ 의료 분야의 직원 중 '간호사' 직업을 가진 데이터만 추출 <br/>
  
  - 직업이 간호사인 데이터만 추출 (Other,Therapist,Administrative,Admin 제외)  1676명 --> 822명 <br/>
    `df = df[df['직무'] == 'Nurse'].reset_index(drop=True)`
    
  ![Image](https://github.com/user-attachments/assets/eebf5d03-3280-44ee-9048-f8861ba70027) 

  #### 2️⃣ 이상치 제외 : 나이가 55세 이상인 간호사  
  
  - 나이로 인해 은퇴하는 것으로 추정  822명 --> 794명 <br/>
  `df = df[df['나이'] < 55].reset_index(drop=True)`

  ![Image](https://github.com/user-attachments/assets/31fc3415-e366-4b25-9672-86aeb1e74172) 

  #### 3️⃣ 수치형 변수만 선택하여 스케일링 (`연령`, `월급`, `근속 연수`)

  &nbsp;&nbsp;&nbsp;&nbsp; `numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns` <br/>
  &nbsp;&nbsp;&nbsp;&nbsp; `scaler = StandardScaler()` <br/>
  &nbsp;&nbsp;&nbsp;&nbsp; `X_scaled_numeric = scaler.fit_transform(X[numeric_cols])` <br/>

  #### 4️⃣ 범주형 변수는 그대로 두고 결합 (`부서`, `직무`, `결혼 상태`)

  &nbsp;&nbsp;&nbsp;&nbsp; `X_encoded = pd.concat([X_scaled_numeric, X.drop(columns=numeric_cols)], axis=1)`

  #### 5️⃣ 피처에 대해서만 원핫 인코딩

  &nbsp;&nbsp;&nbsp;&nbsp; `X_encoded = pd.get_dummies(X_encoded, drop_first=True)`
  
  
### 6. 데이터 분할 및 학습
  `X_train, X_test, y_train, y_test = train_test_split(X_encoded,y,test_size=0.2,random_state=42)`
  #### - 전처리된 훈련 데이터를 이용하여 여러 모델에 대한 학습 진행
  #### - test 데이터의 비율 : 20%
  ![Image](https://github.com/user-attachments/assets/1851311a-3328-4e89-9956-cc124cea0c61) 

  <br/><br/><br/><br/>
### 7. 예측 및 결과 평가 (평가 지표 확인 - Classification_report)

  ![Image](https://github.com/user-attachments/assets/e5812841-c24a-4de0-a55f-83ed33479289)
  ![Image](https://github.com/user-attachments/assets/f47eb4e5-4d34-4ff2-b006-a982d7c97040)
  ![Image](https://github.com/user-attachments/assets/75d80beb-f50e-4c31-8a72-4320f4779cf7)
  ![Image](https://github.com/user-attachments/assets/c18eb1cc-145c-4635-a052-654519048d3e)
  ![Image](https://github.com/user-attachments/assets/84b60bd8-054a-4967-8451-a656a9d98aef)

### 8. 추가 성능 향상 + StratifiedKFold 적용 후 평가 지표
  ![Image](https://github.com/user-attachments/assets/d6604d0e-fbbf-46c6-85c8-db8d5255f0ef) 

1) **gridsearchCV**를 사용하여 **f1-score 기준**으로 최적의 파라미터를 찾음.<br/>
   - Precision과 Recall의 조화평균인 F1-score를 기준으로 최적화함으로써 두 지표 간의 균형을 유지하며, 안정적인 예측 성능을 확보할 수 있도록 함.
2) **데이터의 불균형 문제**(퇴사를 하지 않은 인원이 훨씬 많음)가 있기 때문에 **StratifiedKFold 를 적용<br/>
   - **테스트 시 퇴사인원 비율을 균형있게 유지**하고 **과적합으로 인한 잘못된 성능 예측을 방지** 
  
  ##### 성능 향상 결과 
  ![Image](https://github.com/user-attachments/assets/79d8164e-4ced-41ae-8d58-ead0649b5e05) 

  ### 성능 향상 전/후 Logistic Regression 모델 비교

| Metric | 성능 향상 전<br/> | 성능 향상 후<br/>(StratifiedKFold 적용)
|--------|------------|----------------------|
| **정확도 (Accuracy)** | 0.93 | 0.93 |
| **정밀도 (Precision, Class 1)** | 0.80 | 0.77 |
| **재현율 (Recall, Class 1)** | 0.70 | 0.74 |
| **F1-score (Class 1)** | 0.74 | 0.76 |

**⬇️Precision(정밀도) 감소** → 퇴사자라고 예측했지만 실제 퇴사가 아닌 경우가 소폭 증가<br/>
**⬆️재현율(Recall) 증가** → 퇴사자 탐지 성능 향상<br/>
**⬆️F1-score 상승** → 모델 균형 유지<br/>

∴ Precision 값이 소폭 감소하였으나, **Recall과 F1-score 지표에서 모두 개선**.<br/>
전체적인 모델의 균형과 실질적인 성능(**퇴사자 예측**)을 고려했을 때 미세하더라도 성능 향상이 있다고 판단<br/>
👉🏻 **최종 모델로 선정**
  
  
  #### - 실제 예측 결과
  ![image](https://github.com/user-attachments/assets/86eab15f-f5fb-4433-a0db-d40b716cd04f)

  
  ## ✅학습 결과를 이용한 인사이트 분석
- 여러 모델 중 성능이 좋았던 LightGBM 모델의 가중치를 바탕으로, 퇴사 여부에 큰 영향을 미치는 feature들을 수치와 시각화(그래프)를 통해 확인<br/>
  ![Image](https://github.com/user-attachments/assets/67ceb942-2f9e-4239-b6ba-5091f30c468d) 
  
- 양,음 상관관계 해석을 위해 Logistic Regression 모델 학습 결과에서 추출된 가중치를 이용해 타겟(퇴사여부)에 대한 각 feature의 중요도를 수치로 확인
  ![Image](https://github.com/user-attachments/assets/6da3601a-96ad-484e-a8bd-7c1f52f14774)  <br/><br/>
  ![Image](https://github.com/user-attachments/assets/170c7f4a-aa1a-4507-85a0-b74ac581a44a)

  ### 인사이트
  - '퇴사여부'에 가장 큰 영향을 끼치는 요소는 **'초과근무여부'** 이다.
  - '월급'이 '퇴사여부'에 두번째로 큰 영향을 끼친다.
  - '나이'가 많을수록 퇴사할 확률이 적다.
  - '마지막 승진 이후 근무 연수'가 길수록 퇴사할 확률이 크다.
  - '출퇴근거리'가 길수록 퇴사할 확률이 크다. 

  ### 제안 가능한 이탈 방지 방안
  - 초과근무를 줄이고 워라밸을 개선하는 것이 최우선.
  - 신입 간호사들이 조직에 정착할 수 있도록 초기 적응 프로그램 운영.
  - 급여 및 보상 체계를 조정하여 젊은 간호사들의 만족도를 높이는 것.
  - 출퇴근 부담을 줄이고, 장기적으로 승진 및 경력 개발 기회를 제공하는 것이 중요.
 
  ### 신규/경력 간호사 채용 시 
  - 결혼을 한 상태이며 나이가 어느정도 있고, 병원과 가까운 곳에 거주하는 간호사를 채용하는 것이 유리
