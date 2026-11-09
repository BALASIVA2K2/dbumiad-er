# Datasets (not stored in git)



## ZhangLab (Kermany et al., 2018)

Kaggle path: /kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest\_xray

- train NORMAL: 1,341   train PNEUMONIA: 3,875

- test NORMAL: 234      test PNEUMONIA: 390

- val: 8 NORMAL + 8 PNEUMONIA (not used); validation carved from train NORMAL

- Ignore the nested duplicate "chest\_xray/" folder and "\_\_MACOSX/"; use top-level train/ and test/ only



\## RSNA Pneumonia Detection Challenge

Kaggle path: /kaggle/input/competitions/rsna-pneumonia-detection-challenge

- Total patients: 26,684

- Normal: 8,851   Lung Opacity: 6,012   No Lung Opacity / Not Normal: 11,821

- Bounding boxes: 9,555 (some Lung Opacity images have more than one box)

- Plan: Normal = normal class, Lung Opacity = anomaly; "Not Normal" excluded



## CheXpert (Irvin et al., 2019), small version

Kaggle path: /kaggle/input/datasets/ashery/chexpert

Access: Stanford AIMI research use agreement accepted on \_\_\_ (date)

- CSV "Path" values start with "CheXpert-v1.0-small/"; strip this prefix and join to the Kaggle path

- train.csv: 223,414 rows; frontal 191,027 (No Finding = 16,974, other = 174,053)

- valid.csv: 234 rows; frontal 202 (No Finding = 26, other = 176)

- Plan: frontal only; No Finding = normal; test split built from train.csv with patient-level separation (to be decided)

