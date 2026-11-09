# Datasets (not stored in git)



## ZhangLab (Kermany et al., 2018)

Kaggle path: /kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest\_xray

- train NORMAL: 1,341   train PNEUMONIA: 3,875

- test NORMAL: 234      test PNEUMONIA: 390

- val: 8 NORMAL + 8 PNEUMONIA (not used); validation carved from train NORMAL

- Ignore the nested duplicate "chest\_xray/" folder and "\_\_MACOSX/"; use top-level train/ and test/ only



## RSNA Pneumonia Detection Challenge

Kaggle path: /kaggle/input/competitions/rsna-pneumonia-detection-challenge

- Total patients: 26,684

- Normal: 8,851   Lung Opacity: 6,012   No Lung Opacity / Not Normal: 11,821

- Bounding boxes: 9,555 (some Lung Opacity images have more than one box)

- Plan: Normal = normal class, Lung Opacity = anomaly; "Not Normal" excluded



## CheXpert (Irvin et al., 2019), small version

Kaggle path: /kaggle/input/datasets/ashery/chexpert

Access: Stanford AIMI research use agreement accepted on 08\11\2026 (date)

- CSV "Path" values start with "CheXpert-v1.0-small/"; strip this prefix and join to the Kaggle path

- train.csv: 223,414 rows; frontal 191,027 (No Finding = 16,974, other = 174,053)

- valid.csv: 234 rows; frontal 202 (No Finding = 26, other = 176)

- Plan: frontal only; No Finding = normal; test split built from train.csv with patient-level separation (to be decided)



## Split decisions (fixed)

- One model per dataset; train on that dataset's normal images only; no cross-dataset testing

- Split seed = 42 (separate from training seeds 1-3); split CSVs committed in data/splits/

- Validation used only for hyperparameters and F1 threshold; test used once for final results

- ZhangLab: train 1,200 normal; val 141 normal + 141 pneumonia (from train folder); test official 234 + 390

- RSNA: Normal vs Lung Opacity ("Not Normal" excluded); train 7,551 normal; val 300 + 300; test 1,000 + 1,000; Dice on RSNA only (box masks)

- CheXpert: frontal only; normal = No Finding = 1; abnormal = any pathology = 1 (excluding Support Devices, uncertain-only excluded); patient-level split; val 300 + 300; test 1,000 + 1,000; train = remaining normals