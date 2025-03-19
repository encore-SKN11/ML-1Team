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

  **프로젝트 개요**
- 간호사의 **퇴사 여부(Attrition)** 를 예측하는 머신러닝 모델을 개발하여, 의료기관에서 인력 관리를 효과적으로 수행할 수 있도록 돕는 것이 목표이다.

- **기대효과**:
1. **퇴사 예측을 통한 인력 관리 최적화**
  - 퇴사 가능성이 높은 간호사를 조기에 파악하여 사전 대응 가능
  - 앞으로 고용 시 이탈 가능성이 낮은 특징을 가진 지원자를 선별
2. **근무 환경 및 정책 개선에 활용** : 효과적인 인사·복지 정책을 수립
3. **의료 기관의 운영 효율성 증대** : 인력 부족 문제를 방지함으로써 환자 케어의 질을 유지
4. 인력 채용과 교육에 드는 비용 절감

- **접근 방식**:
  - 목적에 맞는 전처리를 진행하고 여러 모델 중 학습 후 성능이 좋은 모델을 바탕으로, 퇴사 여부에 큰 영향을 미치는 feature들을 수치나 시각화(그래프 등)를 통해 확인하고 이를 통해 병원에 제공할 수 있는 인사이트를 도출
   
- **타겟 변수**: **`퇴사여부`** ('Yes': 퇴사 o , 'No': 퇴사 x)


## 1.데이터 구성 
## - 데이터소스: https://www.kaggle.com/datasets/jpmiller/employee-attrition-for-healthcare/data
- 주요 컬럼:
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

  
### 3. 결측치 및 이상치 탐색
  - 결측치 : **없음**

  - 이상치 : 나이,월급,현 매니저와 근속연수 에서 이상치 발견 <br/>
      → '나이' feature의 이상치 가능성 높음 : ('월급이 적다', '직무에 만족하지 못한다' 등 피처들로 인한 요인보다 나이로 인해 은퇴하는 것으로 추정)
      ![Image](https://github.com/user-attachments/assets/c038dbb8-f705-43ce-ad1c-0420f88ac0d4)


### 4. 데이터 시각화를 통한 탐색

   #### 1️⃣퇴사 여부 분포 : 퇴사한 인원('Yes')과 퇴사하지 않은 인원('No')의 데이터 분포를 나타낸 그래프
   
  ![Image](https://github.com/user-attachments/assets/e16e2910-2214-414d-8652-0305ab6340ef)
  - 퇴사한 인원(Yes)와 퇴사하지 않은 인원(No)의 데이터 분포를 나타낸 그래프
  - 퇴사하지 않은 인원(0)의 수가 퇴사한 인원(1)보다 훨씬 많음 → **데이터 불균형 존재**


   #### 2️⃣ 수치형 변수 간 상관관계 : 지표 간의 상관관계를 나타낸 히트맵(Heatmap)
![Image](https://github.com/user-attachments/assets/f6b37cda-ae48-4db5-a976-914ed16acea6)
  - 주요 지표들이 서로 얼마나 강한 관계를 가지는지 시각적으로 표현
  - 색상(파란색~빨간색)으로 상관계수의 크기를 나타냄
      - 빨간색(1에 가까움): 강한 양의 상관관계 (서로 비례)
      - 파란색(-1에 가까움): 강한 음의 상관관계 (서로 반비례)
      - 흰색(0에 가까움): 거의 상관없음<br/>
  
    #### 3️⃣ 결측치 개수 확인 결과
    
    ![Image](https://github.com/user-attachments/assets/35c77b40-17f7-46af-b8e9-4d2d269a8910) 
  - 모든 컬럼에서 결측치 없음 (0으로 표시됨)
    
   

### 5. 데이터 정제 및 전처리
  #### 1️⃣ 의료 분야의 직원 중 '간호사' 직업을 가진 데이터만 추출 <br/>
  
  - 직업이 간호사인 데이터만 추출 (Other,Therapist,Administrative,Admin 제외)  1676명 --> 822명 <br/>
    `df = df[df['직무'] == 'Nurse'].reset_index(drop=True)`
    
  ![Image](https://github.com/user-attachments/assets/eebf5d03-3280-44ee-9048-f8861ba70027) 

  #### 2️⃣ 이상치 제외 : 나이가 55세 이상인 간호사  
  
  - 나이로 인해 은퇴하는 것으로 추정  822명 --> 794명 <br/>
  `df = df[df['나이'] < 55].reset_index(drop=True)`

  ![Image](https://github.com/user-attachments/assets/31fc3415-e366-4b25-9672-86aeb1e74172) 

  #### 3️⃣ 수치형 변수만 선택하여 스케일링

  &nbsp;&nbsp;&nbsp;&nbsp; `numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns` <br/>
  &nbsp;&nbsp;&nbsp;&nbsp; `scaler = StandardScaler()` <br/>
  &nbsp;&nbsp;&nbsp;&nbsp; `X_scaled_numeric = scaler.fit_transform(X[numeric_cols])` <br/>

  #### 4️⃣ 범주형 변수는 그대로 두고 결합

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

  #### - 실제 예측 결과
  ![image](https://github.com/user-attachments/assets/86eab15f-f5fb-4433-a0db-d40b716cd04f)

  

  ## ✅학습 결과를 이용한 인사이트 분석
- 여러 모델 중 성능이 좋았던 Logistic Regression 모델의 가중치를 바탕으로, 퇴사 여부에 큰 영향을 미치는 feature들을 수치와 시각화(그래프)를 통해 확인하고 이를 통해 병원에 제공할 수 있는 인사이트를 도출
  
  - Logistic Regression 모델 학습 결과에서 추출된 가중치를 이용해 타겟(퇴사여부)에 대한 각 feature의 중요도를 수치로 확인
    
  ![Image](https://github.com/user-attachments/assets/6da3601a-96ad-484e-a8bd-7c1f52f14774)  <br/><br/>
  
  - Logistic Regression 모델 학습 결과에서 추출된 가중치를 이용해 타겟(퇴사여부)에 대한 각 feature의 중요도를 그래프로 확인
  ![Image](https://github.com/user-attachments/assets/170c7f4a-aa1a-4507-85a0-b74ac581a44a)

  ### 인사이트
  - '퇴사여부'에 가장 큰 영향을 끼치는 요소는 **'초과근무여부'** 이다.
  - '결혼여부'가 '퇴사여부'에 두번째로 큰 영향을 끼친다.
  - '나이'가 많을수록 퇴사할 확률이 적다.
  - '마지막 승진 이후 근무 연수'가 길수록 퇴사할 확률이 크다.
  - '출장빈도'가 많을수록 퇴사할 확률이 크다.
  - '출퇴근거리'가 짧을수록 퇴사할 확률이 적다. 

  ### 제안 가능한 이탈 방지 방안
  - 초과근무 줄이기
  - 승진 잘 시켜주기
  - 출장 줄이기
 
  ### 신규/경력 간호사 채용 시 
  - 결혼을 했으며, 나이가 어느정도 있고, 병원과 가까운 곳에 거주하는 인원 채용하는 것이 유리
