
SISTEM KLASIFIKASI ADHD BERBASIS 3D RESNET-50


DESKRIPSI PROYEK:
-----------------
Aplikasi web berbasis Streamlit ini dirancang untuk melakukan klasifikasi 
otomatis ADHD (Attention Deficit Hyperactivity Disorder) menggunakan citra 
structural MRI (sMRI) 3D. Sistem menggunakan arsitektur MedicalNet 
(3D ResNet-50) dengan pendekatan ensemble majority voting dari tiga 
jaringan otak: White Matter (WM), Gray Matter (GM), dan Cerebrospinal Fluid (CSF).


TEKNOLOGI & LIBRARY:
--------------------
- Framework: PyTorch
- Web App: Streamlit
- Medical Imaging: NiBabel, Scipy, NumPy
- Visualization: Plotly, Pillow
- Model: MedicalNet (3D ResNet-50)

STRUKTUR DIREKTORI:
-------------------
/ (Root)
|-- app.py                      (File utama aplikasi)
|-- models/
|   |-- medicalnet/             (Arsitektur model)
|   |-- seg_to_classifier.py    (Wrapper classifier)
|   |-- best_medicalnet_classifier_WM.pth
|   |-- best_medicalnet_classifier_GM.pth
|   |-- best_medicalnet_classifier_CSF.pth
|-- requirements.txt            (Daftar library)
|-- README.txt                  (File ini)

CARA INSTALASI:
---------------
1. Pastikan Python 3.9+ sudah terinstal.
2. Buat Virtual Environment:
   python -m venv venv
3. Aktivasi Virtual Environment:
   (Windows) venv\Scripts\activate
4. Instal Library:
   pip install torch torchvision torchaudio
   pip install streamlit nibabel numpy scipy plotly pillow

CARA MENJALANKAN APLIKASI:
--------------------------
1. Masuk ke folder proyek melalui terminal/cmd.
2. Jalankan perintah:
   streamlit run app.py
3. Buka browser pada alamat yang tertera (biasanya localhost:8501).

ALUR KERJA SISTEM (METODOLOGI):
-------------------------------
1. Preprocessing: Resizing ke 64x64x64 dan Normalisasi Min-Max.
2. Segmentasi: Pemisahan voxel berdasarkan intensitas (WM, GM, CSF).
3. Inference: Prediksi probabilitas per segmen menggunakan ResNet-50.
4. Voting: Penentuan diagnosis akhir (ADHD/TDC) berdasarkan suara terbanyak.


---------------------------------------------------------------------------
DISKLAIMER: 
Sistem ini adalah instrumen penelitian akademik dan bukan merupakan alat 
diagnosis medis berlisensi.
---------------------------------------------------------------------------
=======
# System-Klasifikasi-ADHD---TA
3D MRI Brain Classification for ADHD Detection using MedicalNet (3D ResNet-50) &amp; Ensemble Majority Voting. Features multi-tissue segmentation (WM, GM, CSF) and an interactive Streamlit dashboard. Built with PyTorch.
