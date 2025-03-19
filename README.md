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

- **목적 / 기대효과**:
  본 프로젝트를 통해 간호사 이탈에 영향을 주는 주요 요인을 정량적으로 파악하고, 이를 바탕으로 해당 병원에
  1. 효과적인 인사·복지 정책을 수립
  2. 앞으로 고용 시 이탈 가능성이 낮은 특징을 가진 지원자를 선별
  3. 인력 채용과 교육에 드는 비용 절감
  4. 장기적으로 인사 관리 효율성 증대<br/><br/>
등 다양한 인사이트를 제공함으로써 데이터에 기반한 의사결정을 돕기 위함

- **접근 방식**:
  - 목적에 맞는 전처리를 진행하고 여러 모델 중 학습 후 성능이 좋은 모델을 바탕으로, 퇴사 여부에 큰 영향을 미치는 feature들을 수치나 시각화(그래프 등)를 통해 확인하고 이를 통해 병원에 제공할 수 있는 인사이트를 도출
   
- **타겟 변수**: **`퇴사여부`** ('Yes': 퇴사 o , 'No': 퇴사 x)


## 1.데이터 구성 

### 📂 **연도별 데이터 파일 목록**
각 연도의 외야수 데이터를 개별 CSV 파일로 정리

| 연도  | 데이터 파일명       | 설명 |
|------|----------------|-------------------------------|
| 2000 | `data/2000.csv` | 2000년도 외야수 성적 데이터 |
| 2001 | `data/2001.csv` | 2001년도 외야수 성적 데이터 |
| ...  | ...            | ... |
| 2023 | `data/2023.csv` | 2023년도 외야수 성적 데이터 |
| 2024 | `data/2024.csv` | 2024년도 외야수 성적 데이터 |

### 📊 **연도별 데이터 샘플 ex. '2000.csv'**  
각 연도별 CSV에는 아래와 같은 성적 지표가 포함 

| 이름  | 연도  | WAR  | 득점 | 안타 | 2루타 | 3루타 | 홈런 | 타점 | 도루 | 볼넷 | 사구 | 고의사구 | 삼진 | 병살 | 희생타 | 희생플라이 | 타율  | 출루  | 장타  | OPS   | wRC+  | 수상여부 |
|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| 송지만  | 2000 | 5.61  | 93   | 158  | 33   | 2    | 32   | 90   | 20   | 52   | 7    | 2      | 72   | 10   | 0    | 3    | 0.338 | 0.409 | 0.622 | 1.031 | 169.4  | 1 |
| 박재홍  | 2000 | 5.26  | 101  | 151  | 31   | 5    | 32   | 115  | 30   | 64   | 7    | 5      | 77   | 15   | 0    | 12   | 0.309 | 0.388 | 0.589 | 0.977 | 150.3  | 1 |
| ...    | ...  | ...   | ...  | ...  | ...  | ...  | ...  | ...  | ...  | ...  | ...  | ...    | ...  | ...  | ...  | ...  | ...   | ...   | ...   | ...   | ...    | ... |
| 김응국  | 2000 | 1.03  | 57   | 111  | 15   | 3    | 6    | 45   | 9    | 34   | 2    | 4      | 78   | 7    | 1    | 1    | 0.277 | 0.336 | 0.374 | 0.710 | 86.5   | 0 |
| 채종범  | 2000 | 0.24  | 55   | 113  | 22   | 5    | 8    | 52   | 5    | 35   | 9    | 1      | 59   | 10   | 13   | 4    | 0.247 | 0.311 | 0.370 | 0.681 | 72.0   | 0 |


### 🔹 `yearly_averages_csv`
각 연도별 **주요 성적 지표(8개)** 평균값

| 연도  | WAR  | 타율  | 홈런  | 장타  | OPS  | wRC+  | 안타  | 출루  |
|------|------|------|------|------|------|------|------|------|
| 2000 | 3.264 | 0.293 | 17.833 | 0.479 | 0.847 | 121.872 | 134.056 | 0.368 |
| 2001 | 3.211 | 0.303 | 14.053 | 0.468 | 0.858 | 126.895 | 129.211 | 0.390 |
| 2002 | 3.393 | 0.285 | 15.722 | 0.451 | 0.815 | 123.556 | 125.333 | 0.364 |
| ...  | ...   | ...   | ...   | ...   | ...   | ...   | ...   | ...   |
| 2022 | 3.936 | 0.295 | 13.474 | 0.442 | 0.812 | 130.400 | 149.263 | 0.369 |
| 2023 | 3.006 | 0.290 | 8.765  | 0.412 | 0.773 | 118.294 | 135.118 | 0.360 |
| 2024 | 2.879 | 0.299 | 14.737 | 0.453 | 0.830 | 116.368 | 147.421 | 0.377 |


## 🧹 EDA

### 1. 데이터 로드

![Image](https://github.com/user-attachments/assets/b68ae529-31d4-42e5-8ab4-255d80161843)
  
  **연도별 개별 CSV → 하나의 통합 CSV 파일 생성**
  - `glob.glob("data/*.csv")` 를 이용해 data 폴더 내 모든 CSV 파일을 가져옴 → **'all.csv'** 파일에 통합하여 저장


### 2. 데이터 구조 및 기초 통계 확인
![Image](https://github.com/user-attachments/assets/57e37e6d-1dbc-4f51-afad-04b9a74cfcaf)

  
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
