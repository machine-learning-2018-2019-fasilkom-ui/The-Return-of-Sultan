# The-Return-of-Sultan

# Deskripsi Kelompok
Nama Kelompok : The Return Of Sultan <br/>
Title : [Komparasi Decision Tree dan Random Forest untuk Klasifikasi Kapal pada Citra Satelit di Wilayah Maritim](Documents/The%20Return%20of%20Sultan_proposal.pdf) <br/>

Dataset dapat di download pada situs [Kaggle - Ship in Satellite Imagery](https://www.kaggle.com/rhammell/ships-in-satellite-imagery) <br/>

# Progress 1
Ekstraksi Fitur menggunakan Histogram of Oriented Gradient (HOG).

|  | Kapal | Bukan Kapal |
| --- | --- | --- |
| Citra Satelit | ![Alt text](images/kapal1.jpg?raw=true "HOG Kapal") | ![Alt text](images/bukankapal1.jpg?raw=true "HOG Bukan Kapal") |
| Fitur HOG | ![Alt text](images/kapal1-hog-or_8_cell_4-4.jpg?raw=true "HOG Kapal") | ![Alt text](images/bukankapal1-hog-or_8_cell_4-4.jpg?raw=true "HOG Bukan Kapal") |

Untuk dokumen lengkap dapat dilihat pada dokumen [Progress 1](Documents/The%20Return%20of%20Sultan_progress1.pdf).

# Progress 2
Proses Klasifikasi menggunakan Algoritma Decision Tree dan Random Forest. Untuk algoritma Decision Tree terbagi menjadi dua jenis yaitu versi tanpa library dan menggunakan library standar machine learning (Scikit). Untuk menjalankan kode dapat menjalankannya dengan 5 file kode python sebagai berikut: 
1. [main_hog_dt_kfold.py](main_hog_dt_kfold.py) - Kode Klasifikasi Decision Tree dengan Stratified K-Fold Cross Validation
2. [main_hog_dt_manual.py](main_hog_dt_manual.py) - Kode Klasifikasi Decision Tree (Implementasi Manual Tanpa Library)
3. [main_hog_dt_non_fold.py](main_hog_dt_non_fold.py) - Kode Klasifikasi Decision Tree dengan Pembagian data latih 80% dan data uji 20%
4. [main_hog_rf_kfold.py](main_hog_rf_kfold.py) - Kode Klasifikasi Random Forest dengan Stratified K-Fold Cross Validation
5. [main_hog_rf_non_fold.py](main_hog_rf_non_fold.py) - Kode Klasifikasi Random Forest dengan Pembagian data latih 80% dan data uji 20%

Untuk dokumen lengkap dapat dilihat pada dokumen [Progress 2](Documents/The%20Return%20of%20Sultan_progress2.pdf).

# Data Augmentation, Ensemble Method, and Final Report & Poster
1. [main_augmentation.py](main_augmentation.py) - Kode untuk proses augmentasi citra kapal dan bukan kapal yang terdiri dari proses rotasi, horizontal flip, vertical flip, gaussian noise, salt & pepper noise, blurring, rescale

| Rotate 15 | Rotate 30 | Rotate 45 | Rotate 60 | Rotate 75 | Rotate 90 | Rotate 105 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ![Alt text](images/20160710_182139_0c78-rotate15.png?raw=true "Rotate 15") | ![Alt text](images/20160710_182139_0c78-rotate30.png?raw=true "Rotate 30") | ![Alt text](images/20160710_182139_0c78-rotate45.png?raw=true "Rotate 45") | ![Alt text](images/20160710_182139_0c78-rotate60.png?raw=true "Rotate 60") | ![Alt text](images/20160710_182139_0c78-rotate75.png?raw=true "Rotate 75") | ![Alt text](images/20160710_182139_0c78-rotate90.png?raw=true "Rotate 90") | ![Alt text](images/20160710_182139_0c78-rotate105.png?raw=true "Rotate 105") |
| Rotate 120 | Rotate 135 | Rotate 150 | Rotate 165 | Rotate 180 | Rotate 195 | Rotate 210 |
| ![Alt text](images/20160710_182139_0c78-rotate120.png?raw=true "Rotate 120") | ![Alt text](images/20160710_182139_0c78-rotate135.png?raw=true "Rotate 135") | ![Alt text](images/20160710_182139_0c78-rotate150.png?raw=true "Rotate 150") | ![Alt text](images/20160710_182139_0c78-rotate165.png?raw=true "Rotate 165") | ![Alt text](images/20160710_182139_0c78-rotate180.png?raw=true "Rotate 180") | ![Alt text](images/20160710_182139_0c78-rotate195.png?raw=true "Rotate 195") | ![Alt text](images/20160710_182139_0c78-rotate210.png?raw=true "Rotate 210") |
| Rotate 225 | Rotate 240 | Rotate 255 | Rotate 270 | Rotate 285 | Rotate 300 | Rotate 315 |
| ![Alt text](images/20160710_182139_0c78-rotate225.png?raw=true "Rotate 225") | ![Alt text](images/20160710_182139_0c78-rotate240.png?raw=true "Rotate 240") | ![Alt text](images/20160710_182139_0c78-rotate255.png?raw=true "Rotate 255") | ![Alt text](images/20160710_182139_0c78-rotate270.png?raw=true "Rotate 270") | ![Alt text](images/20160710_182139_0c78-rotate285.png?raw=true "Rotate 285") | ![Alt text](images/20160710_182139_0c78-rotate300.png?raw=true "Rotate 300") | ![Alt text](images/20160710_182139_0c78-rotate315.png?raw=true "Rotate 315") |
| Rotate 330 | Rotate 345 | Rotate 360 |
| ![Alt text](images/20160710_182139_0c78-rotate330.png?raw=true "Rotate 330") | ![Alt text](images/20160710_182139_0c78-rotate345.png?raw=true "Rotate 345") | ![Alt text](images/20160710_182139_0c78-rotate360.png?raw=true "Rotate 360") |

| Salt & Pepper Noise | Gaussian Noise |
| :---: | :---: |
| ![Alt text](images/20160710_182139_0c78-s&p%20noise.png?raw=true "Salt & Pepper Noise") | ![Alt text](images/20160710_182139_0c78-gaussian%20noise.png?raw=true "Gaussian Noise") |

| Horizontal Flip | Vertical Flip |
| :---: | :---: |
| ![Alt text](images/20160710_182139_0c78-HFlip.png?raw=true "Horizontal Flip") | ![Alt text](images/20160710_182139_0c78-VFlip.png?raw=true "Vertical Flip") |

| Blurring (3x3) | Blurring (5x5) |
| :---: | :---: |
| ![Alt text](images/20160710_182139_0c78-VerySoft%20Blur.png?raw=true "Blurring (3x3)") | ![Alt text](images/20160710_182139_0c78-Soft%20blur.png?raw=true "Blurring (5x5)") |

| Rescale (0.75) |
| :---: |
| ![Alt text](images/20160710_182139_0c78-Rescale_0.75.png?raw=true "Rescale (0.75)") | 

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