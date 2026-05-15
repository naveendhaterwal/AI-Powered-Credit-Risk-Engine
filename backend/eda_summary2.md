## 8. Categorical Feature Audit
### loan_limit
- Unique values: 3
  - cf: 135348 (91.04%)
  - ncf: 9978 (6.71%)
  - nan: 3344 (2.25%)

### Gender
- Unique values: 4
  - Male: 42346 (28.48%)
  - Joint: 41399 (27.85%)
  - Sex Not Available: 37659 (25.33%)
  - Female: 27266 (18.34%)

### approv_in_adv
- Unique values: 3
  - nopre: 124621 (83.82%)
  - pre: 23141 (15.57%)
  - nan: 908 (0.61%)

### loan_type
- Unique values: 3
  - type1: 113173 (76.12%)
  - type2: 20762 (13.97%)
  - type3: 14735 (9.91%)

### loan_purpose
- Unique values: 5
  - p3: 55934 (37.62%)
  - p4: 54799 (36.86%)
  - p1: 34529 (23.23%)
  - p2: 3274 (2.20%)
  - nan: 134 (0.09%)

### Credit_Worthiness
- Unique values: 2
  - l1: 142344 (95.74%)
  - l2: 6326 (4.26%)

### open_credit
- Unique values: 2
  - nopc: 148114 (99.63%)
  - opc: 556 (0.37%)

### business_or_commercial
- Unique values: 2
  - nob/c: 127908 (86.03%)
  - b/c: 20762 (13.97%)

### Neg_ammortization
- Unique values: 3
  - not_neg: 133420 (89.74%)
  - neg_amm: 15129 (10.18%)
  - nan: 121 (0.08%)

### interest_only
- Unique values: 2
  - not_int: 141560 (95.22%)
  - int_only: 7110 (4.78%)

### lump_sum_payment
- Unique values: 2
  - not_lpsm: 145286 (97.72%)
  - lpsm: 3384 (2.28%)

### construction_type
- Unique values: 2
  - sb: 148637 (99.98%)
  - mh: 33 (0.02%)

### occupancy_type
- Unique values: 3
  - pr: 138201 (92.96%)
  - ir: 7340 (4.94%)
  - sr: 3129 (2.10%)

### Secured_by
- Unique values: 2
  - home: 148637 (99.98%)
  - land: 33 (0.02%)

### total_units
- Unique values: 4
  - 1U: 146480 (98.53%)
  - 2U: 1477 (0.99%)
  - 3U: 393 (0.26%)
  - 4U: 320 (0.22%)

### credit_type
- Unique values: 4
  - CIB: 48152 (32.39%)
  - CRIF: 43901 (29.53%)
  - EXP: 41319 (27.79%)
  - EQUI: 15298 (10.29%)

### co-applicant_credit_type
- Unique values: 2
  - CIB: 74392 (50.04%)
  - EXP: 74278 (49.96%)

### age
- Unique values: 8
  - 45-54: 34720 (23.35%)
  - 35-44: 32818 (22.07%)
  - 55-64: 32534 (21.88%)
  - 65-74: 20744 (13.95%)
  - 25-34: 19142 (12.88%)

### submission_of_application
- Unique values: 3
  - to_inst: 95814 (64.45%)
  - not_inst: 52656 (35.42%)
  - nan: 200 (0.13%)

### Region
- Unique values: 4
  - North: 74722 (50.26%)
  - south: 64016 (43.06%)
  - central: 8697 (5.85%)
  - North-East: 1235 (0.83%)

### Security_Type
- Unique values: 2
  - direct: 148637 (99.98%)
  - Indriect: 33 (0.02%)

## Target Correlation
income                 -0.065119
property_value         -0.048864
loan_amount            -0.036825
Upfront_charges        -0.019138
term                   -0.000240
ID                      0.001703
Credit_Score            0.004004
rate_of_interest        0.022957
LTV                     0.038895
dtir1                   0.078083
Status                  1.000000
year                         NaN
Interest_rate_spread         NaN
Name: Status, dtype: float64