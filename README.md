# The-Return-of-Sultan

# Deskripsi Kelompok
Nama Kelompok : The Return Of Sultan <br/>
Title : [Komparasi Decision Tree dan Random Forest untuk Klasifikasi Kapal pada Citra Satelit di Wilayah Maritim](Documents/The%20Return%20of%20Sultan_proposal.pdf) <br/>

Dataset dapat di download pada situs [Kaggle - Ship in Satellite Imagery](https://www.kaggle.com/rhammell/ships-in-satellite-imagery) <br/>

# Progress 1
Ekstraksi Fitur menggunakan Histogram of Oriented Gradient (HOG).

|  | Kapal | Bukan Kapal |
| --- | --- | --- |
| Citra Satelit | ![Alt text](Images/kapal1.JPG?raw=true "HOG Kapal") | ![Alt text](Images/bukankapal1.JPG?raw=true "HOG Bukan Kapal") |
| Fitur HOG | ![Alt text](Images/kapal1-hog-or_8_cell_4-4.JPG?raw=true "HOG Kapal") | ![Alt text](Images/bukankapal1-hog-or_8_cell_4-4.JPG?raw=true "HOG Bukan Kapal") |

Untuk dokumen lengkap dapat dilihat pada dokumen [Progress 1](Documents/The%20Return%20of%20Sultan_progress1.pdf).

# Progress 2
Proses Klasifikasi menggunakan Algoritma Decision Tree dan Random Forest. Untuk algoritma Decision Tree terbagi menjadi dua jenis yaitu versi tanpa library dan menggunakan library standar machine learning (Scikit). Untuk menjalankan kode dapat menjalankannya dengan 5 file kode python sebagai berikut: 
1. [main_hog_dt_kfold.py](main_hog_dt_kfold.py) - Kode Klasifikasi Decision Tree dengan Stratified K-Fold Cross Validation
2. [main_hog_dt_manual.py](main_hog_dt_manual.py) - Kode Klasifikasi Decision Tree (Implementasi Manual Tanpa Library)
3. [main_hog_dt_non_fold.py](main_hog_dt_non_fold.py) - Kode Klasifikasi Decision Tree dengan Pembagian data latih 80% dan data uji 20%
4. [main_hog_rf_kfold.py](main_hog_rf_kfold.py) - Kode Klasifikasi Random Forest dengan Stratified K-Fold Cross Validation
5. [main_hog_rf_non_fold.py](main_hog_rf_non_fold.py) - Kode Klasifikasi Random Forest dengan Pembagian data latih 80% dan data uji 20%

Untuk dokumen lengkap dapat dilihat pada dokumen [Progress 2](Documents/The%20Return%20of%20Sultan_progress2.pdf).

# Data Augmentation and Ensemble Method
1. [main_augmentation.py](main_augmentation.py) - Kode untuk proses augmentasi citra kapal dan bukan kapal yang terdiri dari proses horizontal flip, vertical flip, gaussian noise, salt & pepper noise, blurring
2. [main_hog_dt_5_fold_aug.py](main_hog_dt_5_fold_aug.py) - Kode klasifikasi Decision Tree menggunakan data augmentasi dan stratified k-fold cross validation = 5
3. [main_hog_dt_10_fold_aug.py](main_hog_dt_10_fold_aug.py) - Kode klasifikasi Decision Tree menggunakan data augmentasi dan stratified k-fold cross validation = 10
4. [main_hog_dt_5_fold.py](main_hog_dt_5_fold.py) - Kode klasifikasi Decision Tree dengan stratified k-fold cross validation = 5
5. [main_hog_dt_10_fold.py](main_hog_dt_10_fold.py) - Kode klasifikasi Decision Tree dengan stratified k-fold cross validation = 10
6. [main_hog_dt_aug.py](main_hog_dt_aug.py) - Kode klasifikasi Decision Tree dengan data augmentasi
7. [main_hog_dt_tuning_grid_aug.py](main_hog_dt_tuning_grid_aug.py) - Kode klasifikasi Decision Tree dengan menggunakan tuning = grid search dan data augmentasi
8. [main_hog_dt_tuning_grid.py](main_hog_dt_tuning_grid.py) - Kode klasifikasi Decision Tree dengan menggunakan tuning = grid search
9. [main_hog_dt_tuning_random_aug.py](main_hog_dt_tuning_random_aug.py) - Kode klasifikasi Decision Tree dengan menggunakan tuning = random dan data augmentasi
10. [main_hog_dt_tuning_random.py](main_hog_dt_random_grid.py) - Kode klasifikasi Decision Tree dengan menggunakan tuning = random
11. [main_hog_rf_5_fold_aug.py](main_hog_rf_5_fold_aug.py) - Kode klasifikasi Random Forest dengan menggunakan data augmentasi dan stratified k-fold cross validation = 5
12. [main_hog_rf_5_fold.py](main_hog_rf_5_fold.py) - Kode klasifikasi Random Forest dengan stratified k-fold cross validation = 5
13. [main_hog_rf_10_fold_aug.py](main_hog_rf_10_fold_aug.py) - Kode klasifikasi Random Forest dengan menggunakan data augmentasi dan stratified k-fold cross validation = 10
14. [main_hog_rf_10_fold.py](main_hog_rf_10_fold.py) - Kode klasifikasi Random Forest dengan stratified k-fold cross validation = 10
15. [main_hog_rf_aug.py](main_hog_rf_aug.py) - Kode klasifikasi Random Forest dengan data augmentasi
16. [main_hog_rf.py](main_hog_rf.py) - Kode klasifikasi Random Forest
17. [main_xgboost_dt_5_fold_aug.py](main_xgboost_dt_5_fold_aug.py) - Kode klasifikasi Decision tree menggunakan gradient boosting dan data augmentasi serta stratified k-fold cross validation = 5
18. [main_xgboost_dt_5_fold.py](main_xgboost_dt_5_fold.py) - Kode klasifikasi Decision tree menggunakan gradient boosting dan stratified k-fold cross validation = 5
19. [main_xgboost_dt_10_fold_aug.py](main_xgboost_dt_10_fold_aug.py) - Kode klasifikasi Decision tree menggunakan gradient boosting dan data augmentasi serta stratified k-fold cross validation = 10
20. [main_xgboost_dt_10_fold.py](main_xgboost_dt_10_fold.py) - Kode klasifikasi Decision tree menggunakan gradient boosting dan stratified k-fold cross validation = 10
21. [main_xgboost_dt_aug.py](main_xgboost_dt_aug.py) - Kode klasifikasi Decision tree menggunakan gradient boosting dan data augmentasi
22. [main_xgboost_dt.py](main_xgboost_dt.py) - Kode klasifikasi Decision tree menggunakan gradient boosting

| Salt & Pepper Noise | Gaussian Noise |
| :---: | :---: |
| ![Alt text](Images/20160710_182139_0c78-s&p%20noise.png?raw=true "Salt & Pepper Noise") | ![Alt text](Images/20160710_182139_0c78-gaussian%20noise.png?raw=true "Gaussian Noise") |

| Horizontal Flip | Vertical Flip |
| :---: | :---: |
| ![Alt text](Images/20160710_182139_0c78-HFlip.png?raw=true "Horizontal Flip") | ![Alt text](Images/20160710_182139_0c78-VFlip.png?raw=true "Vertical Flip") |

| Blurring (3x3) | Blurring (5x5) |
| :---: | :---: |
| ![Alt text](Images/20160710_182139_0c78-VerySoft%20Blur.png?raw=true "Blurring (3x3)") | ![Alt text](Images/20160710_182139_0c78-Soft%20blur.png?raw=true "Blurring (5x5)") |

| Hasil deteksi dengan decision tree menggunakan gradient boosting pada citra pelabuhan makassar |
| :---: |
| ![Alt text](Hasil%20Eksperimen/dt_xgboost/makasar_1_xgboost.PNG?raw=true "DT Gradient Boosting (Makassar)") |

|  Hasil deteksi dengan random forest menggunakan data augmentasi pada citra pelabuhan makassar |
| :---: |
| ![Alt text](Hasil%20Eksperimen/rf_aug_10fold/makasar_1_rf_10_fold_aug.JPG?raw=true "RF dengan data augmentasi 10-fold (Makassar)") |

# Final Report
Untuk dokumen lengkap dapat dilihat pada dokumen [Final Report](Documents/The%20Return%20of%20Sultan_finalreport.pdf). <br/>

# Banner
![Alt text](Images/Banner3%20Proyek%20ML-%20The%20Return%20of%20Sultan.png?raw=true "Banner - The Return of Sultan") <br/>

# Referensi
[1] Kaggle, “Ships in Satellite Imagery,” 2018. [Online]. Available: https://www.kaggle.com/rhammell/ships-in-satelliteimagery. <br/>
[2] B. P. S. INDONESIA, “STATISTIK SUMBER DAYA LAUT DAN PESISIR.” 2018. <br/>
[3] S. K. R. INDONESIA, “Potensi Besar Perikanan Tangkap Indonesia.” 2016. <br/>
[4] Detik, “Menteri Susi: Kerugian Akibat Illegal Fishing Rp 240 Triliun. 2014. <br/>
[5] Katadata, “Cek Data: Benarkah 488 Kapal Illegal Fishing Sudah Ditenggelamkan?” 2019. <br/>
[6] S. Marsland, Machine Learning: An Algorithmic Perspective, Second Edition, 2nd ed. Chapman & Hall/CRC, 2014. <br/>
[7] Y. Liu, Y. Wang, and J. Zhang, “New Machine Learning Algorithm: Random Forest,” 2012, pp. 246–252. <br/>
[8] Y. Wang, X. Zhu, and B. Wu, “Automatic detection of individual oil palm trees from UAV images using HOG features and an SVM classifier,” vol. 1161, 2018. <br/>
[9] H. Zhou, Y. Zhuang, L. Chen, and H. Shi, “Ship Detection in Optical Satellite Images,” vol. 3, 2018 <br/>