# The-Return-of-Sultan

# Deskripsi Kelompok
Nama Kelompok : The Return Of Sultan
Title : [Komparasi Decision Tree dan Random Forest untuk Klasifikasi Kapal pada Citra Satelit di Wilayah Maritim](Documents/The%20Return%20of%20Sultan_proposal.pdf)

Dataset dapat di download pada situs [Kaggle - Ship in Satellite Imagery](https://www.kaggle.com/rhammell/ships-in-satelliteimagery)

# Progress 1
Ekstraksi Fitur menggunakan Histogram of Oriented Gradient (HOG).

Untuk dokumen lengkap dapat dilihat pada dokumen [Progress 1](Documents/The%20Return%20of%20Sultan_progress1.pdf).

# Progress 2
Proses Klasifikasi menggunakan Algoritma Decision Tree dan Random Forest. Untuk algoritma Decision Tree terbagi menjadi dua jenis yaitu versi tanpa library dan menggunakan library standar machine learning (Scikit). Untuk menjalankan kode dapat menjalankannya dengan 5 file kode python sebagai berikut: 
1. [main_hog_dt_kfold.py](main_hog_dt_kfold.py) - Kode Klasifikasi Decision Tree dengan Stratified K-Fold Cross Validation
2. [main_hog_dt_manual.py](main_hog_dt_manual.py) - Kode Klasifikasi Decision Tree (Implementasi Manual Tanpa Library)
3. [main_hog_dt_non_fold.py](main_hog_dt_non_fold.py) - Kode Klasifikasi Decision Tree dengan Pembagian data latih 80% dan data uji 20%
4. [main_hog_rf_kfold.py](main_hog_rf_kfold.py) - Kode Klasifikasi Random Forest dengan Stratified K-Fold Cross Validation
5. [main_hog_rf_non_fold.py](main_hog_rf_non_fold.py) - Kode Klasifikasi Random Forest dengan Pembagian data latih 80% dan data uji 20%

Untuk dokumen lengkap dapat dilihat pada dokumen [Progress 2](Documents/The%20Return%20of%20Sultan_progress2.pdf).

# Referensi
[1] Kaggle, “Ships in Satellite Imagery,” 2018. [Online]. Available: https://www.kaggle.com/rhammell/ships-in-satelliteimagery.
[2] B. P. S. INDONESIA, “STATISTIK SUMBER DAYA LAUT DAN PESISIR.” 2018.
[3] S. K. R. INDONESIA, “Potensi Besar Perikanan Tangkap Indonesia.” 2016.
[4] Detik, “Menteri Susi: Kerugian Akibat Illegal Fishing Rp 240 Triliun. 2014.
[5] Katadata, “Cek Data: Benarkah 488 Kapal Illegal Fishing Sudah Ditenggelamkan?” 2019.
[6] S. Marsland, Machine Learning: An Algorithmic Perspective, Second Edition, 2nd ed. Chapman & Hall/CRC, 2014.
[7] Y. Liu, Y. Wang, and J. Zhang, “New Machine Learning Algorithm: Random Forest,” 2012, pp. 246–252.
[8] Y. Wang, X. Zhu, and B. Wu, “Automatic detection of individual oil palm trees from UAV images using HOG features and an SVM classifier,” vol. 1161, 2018.
[9] H. Zhou, Y. Zhuang, L. Chen, and H. Shi, “Ship Detection in Optical Satellite Images,” vol. 3, 2018