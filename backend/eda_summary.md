# Credit Risk Dataset Quality Audit & Diagnostic Report

## 1. Dataset Overview
- **Rows:** 148,670
- **Columns:** 34
- **Target Column:** Status
- **Memory Usage:** 173.90 MB
- **Duplicate Rows:** 0

## 2. Missing Value Analysis
- **loan_limit**: 3,344 (2.25%)
- **approv_in_adv**: 908 (0.61%)
- **loan_purpose**: 134 (0.09%)
- **rate_of_interest**: 36,439 (24.51%)
- **Interest_rate_spread**: 36,639 (24.64%)
- **Upfront_charges**: 39,642 (26.66%)
- **term**: 41 (0.03%)
- **Neg_ammortization**: 121 (0.08%)
- **property_value**: 15,098 (10.16%)
- **income**: 9,150 (6.15%)
- **age**: 200 (0.13%)
- **submission_of_application**: 200 (0.13%)
- **LTV**: 15,098 (10.16%)
- **dtir1**: 24,121 (16.22%)


## 3. Target Variable Analysis
Class distribution for `Status`:
- 0: 112,031 (75.36%)
- 1: 36,639 (24.64%)


## 4. Financial Feature Audit
### loan_amount
- Min: 16500.0
- Max: 3576500.0
- Mean: 331117.7439967714
- 1st pctl: 66500.0
- 99th pctl: 856500.0
- Skew: 1.6669980938622415
- Kurtosis: 9.127775255514376
- Zeros: 0 | Negatives: 0

### rate_of_interest
- Min: 0.0
- Max: 8.0
- Mean: 4.045475804367777
- 1st pctl: 2.875
- 99th pctl: 5.5
- Skew: 0.3884060270841701
- Kurtosis: 0.34456403501357347
- Zeros: 1 | Negatives: 0

### Interest_rate_spread
- Min: -3.638
- Max: 3.357
- Mean: 0.4416556604868295
- 1st pctl: -0.6811
- 99th pctl: 1.6196299999999972
- Skew: 0.28076233013056445
- Kurtosis: -0.18356607863153585
- Zeros: 9 | Negatives: 21883

### Upfront_charges
- Min: 0.0
- Max: 60000.0
- Mean: 3224.996126591334
- 1st pctl: 0.0
- 99th pctl: 14297.049899999964
- Skew: 1.7540756791547982
- Kurtosis: 6.368586300587068
- Zeros: 20770 | Negatives: 0

### term
- Min: 96.0
- Max: 360.0
- Mean: 335.1365816899797
- 1st pctl: 180.0
- 99th pctl: 360.0
- Skew: -2.1748217958156495
- Kurtosis: 3.1732363170669395
- Zeros: 0 | Negatives: 0

### property_value
- Min: 8000.0
- Max: 16508000.0
- Mean: 497893.46569640347
- 1st pctl: 88000.0
- 99th pctl: 1808000.0
- Skew: 4.586275832462365
- Kurtosis: 73.22119583882989
- Zeros: 0 | Negatives: 0

### income
- Min: 0.0
- Max: 578580.0
- Mean: 6957.338876146789
- 1st pctl: 600.0
- 99th pctl: 26640.0
- Skew: 17.307695079866924
- Kurtosis: 885.2924596801089
- Zeros: 1260 | Negatives: 0

### Credit_Score
- Min: 500.0
- Max: 900.0
- Mean: 699.7891033833322
- 1st pctl: 504.0
- 99th pctl: 897.0
- Skew: 0.004766756957725898
- Kurtosis: -1.202649443264979
- Zeros: 0 | Negatives: 0

### LTV
- Min: 0.967478198
- Max: 7831.25
- Mean: 72.74645733387138
- 1st pctl: 19.61009174
- 99th pctl: 102.7597403
- Skew: 120.61533746802151
- Kurtosis: 19979.044666198046
- Zeros: 0 | Negatives: 0

### dtir1
- Min: 5.0
- Max: 61.0
- Mean: 37.73293242017198
- 1st pctl: 8.0
- 99th pctl: 60.0
- Skew: -0.5514649624329434
- Kurtosis: 0.37888255574493
- Zeros: 0 | Negatives: 0

## 7. Correlation Analysis
- loan_amount & property_value: 0.73
- rate_of_interest & Interest_rate_spread: 0.61