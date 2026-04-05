import os
import tempfile
import io

import streamlit as st

import torch
import torch.nn as nn

import nibabel as nib
import numpy as np
from scipy import ndimage

from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# ===== MedicalNet (3D ResNet) =====
from models.medicalnet.model import generate_model

from models.medicalnet.model import generate_model
from models.seg_to_classifier import SegToClassifier


# ===== MedicalNet OPT =====
class Opt:
    pass

def build_opt():
    opt = Opt()
    opt.model = 'resnet'
    opt.model_depth = 50
    opt.input_D = 64
    opt.input_H = 64
    opt.input_W = 64
    opt.resnet_shortcut = 'B'
    opt.no_cuda = True
    opt.pretrain_path = ''
    opt.new_layer_names = []
    opt.phase = 'test'
    opt.n_seg_classes = 2
    opt.n_input_channels = 1
    opt.gpu_id = []
    return opt


BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_DIR = os.path.join(BASE_DIR, "models")

# ==================== KONFIGURASI HALAMAN ====================
st.set_page_config(
    page_title="Sistem Klasifikasi ADHD",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        
        /* Pengaturan Layout */
        height: 160px;          /* Tinggi dikurangi sedikit agar lebih kompak */
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 2px;               /* Jarak antar elemen (H3, H1, P) dikunci di sini */
    }

    .metric-card h3 {
        margin: 0 !important;
        padding: 0 !important;
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 400;
        line-height: 1.1;       
    }

    .metric-card h1 {
        margin: 0 !important;
        padding: 0 !important;
        font-size: 2.2rem;      
        font-weight: 700;
        line-height: 1;         
    }

    .metric-card p {
        margin: 0 !important;
        padding: 0 !important;
        font-size: 0.8rem;
        opacity: 0.8;
        line-height: 1.1;
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    
    .segment-label {
        font-weight: 600;
        font-size: 1.1rem;
        color: #333;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ==================== FUNGSI PREPROCESSING ====================
def load_nifti_file(uploaded_file):
    """Load file NIfTI dari uploaded file dengan penanganan buffer yang benar"""
    # Gunakan suffix yang sesuai dengan file asli
    suffix = '.nii.gz' if uploaded_file.name.endswith('.gz') else '.nii'
    
    # delete=False sangat penting agar file tidak hilang saat ditutup sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Load menggunakan path file temporary
        img = nib.load(tmp_path)
        # Force loading data ke memory sebelum file dihapus
        data = img.get_fdata()
    finally:
        # Pastikan file temporary dihapus setelah data masuk ke variabel
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            
    return data

def normalize_volume(volume):
    """Normalisasi volume 3D"""
    volume = volume.astype(np.float32)
    # Normalisasi ke range [0, 1]
    if volume.max() > 0:
        volume = (volume - volume.min()) / (volume.max() - volume.min())
    return volume

def segment_tissue(volume):
    """
    Segmentasi jaringan otak menjadi WM, GM, CSF
    Menggunakan thresholding sederhana (bisa diganti dengan metode yang lebih advanced)
    """
    normalized = normalize_volume(volume)
    
    # Threshold untuk segmentasi (nilai ini bisa disesuaikan)
    # CSF: intensitas rendah
    # GM: intensitas menengah
    # WM: intensitas tinggi
    
    csf_mask = (normalized < 0.3)
    wm_mask = (normalized > 0.7)
    gm_mask = ~(csf_mask | wm_mask)
    
    csf_segment = volume * csf_mask
    gm_segment = volume * gm_mask
    wm_segment = volume * wm_mask
    
    return {
        'WM': wm_segment,
        'GM': gm_segment,
        'CSF': csf_segment
    }


# ==================== FUNGSI PREDIKSI ====================
@st.cache_resource
def load_models(model_paths):
    models_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = build_opt()

    for name, path in model_paths.items():
        if not os.path.exists(path):
            st.sidebar.error(f"⚠️ File tidak ditemukan: {path}")
            continue
            
        try:
            # 1. Bangun base model
            seg_net, _ = generate_model(opt)

            # 2. Bungkus dengan class classifier
            model = SegToClassifier(seg_net).to(device)

            # 3. Load state dict
            state = torch.load(path, map_location=device)
            
            # --- PROSES PEMBERSIHAN KEY (PENTING!) ---
            from collections import OrderedDict
            new_state = OrderedDict()
            
            for k, v in state.items():
                # Hapus prefix 'module.' (jika ada dari DataParallel)
                name_key = k.replace('module.', '')
                # Hapus prefix 'backbone.' (karena error terminal Anda menunjukkan ini)
                name_key = name_key.replace('backbone.', '')
                
                new_state[name_key] = v
            # ------------------------------------------

            # 4. Masukkan state yang sudah dibersihkan
            model.load_state_dict(new_state)
            model.eval()
            
            models_dict[name] = model
            st.sidebar.success(f"✅ {name} model loaded!")
            
        except Exception as e:
            st.sidebar.error(f"❌ Gagal memuat {name}: {str(e)}")
            import traceback
            traceback.print_exc()

    return models_dict, device

# prepocessing
def preprocess_volume_3d(volume):
    volume = volume.astype(np.float32)
    if volume.max() > 0:
        volume = (volume - volume.min()) / (volume.max() - volume.min())

    volume = ndimage.zoom(
        volume,
        (
            64 / volume.shape[0],
            64 / volume.shape[1],
            64 / volume.shape[2]
        ),
        order=1
    )

    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
    return tensor


def majority_voting(predictions_dict):
    """Majority voting dengan perhitungan akurasi rata-rata dari pemenang"""
    all_predictions = []
    all_confidences = []
    
    # Ambil nilai prediksi dan confidence
    for segment_name, (preds, confs) in predictions_dict.items():
        if preds:
            all_predictions.append(preds[0])
            all_confidences.append(confs[0])
    
    if not all_predictions:
        return None, 0.0, np.array([0, 0]), 0.0
    
    # Hitung voting
    votes = np.bincount(all_predictions, minlength=2)
    final_prediction = np.argmax(votes)
    
    # Hitung Ratio Voting (misal 2 dari 3 model setuju)
    vote_ratio = votes[final_prediction] / len(all_predictions)
    
    # HITUNG RATA-RATA CONFIDENCE DARI PEMENANG SAJA
    winning_confidences = [
        conf for pred, conf in zip(all_predictions, all_confidences) 
        if pred == final_prediction
    ]
    avg_winning_confidence = np.mean(winning_confidences) if winning_confidences else 0.0
    
    return final_prediction, vote_ratio, votes, avg_winning_confidence

# ==================== VISUALISASI ====================
@st.cache_data(show_spinner=False)
def visualize_3d_brain(volume, title="Brain MRI 3D View"):
    """Visualisasi 3D brain volume"""
    # Downsample untuk performa
    volume_small = volume[::4, ::4, ::4]
    
    # Threshold untuk visualisasi
    threshold = np.percentile(volume_small[volume_small > 0], 50)
    
    X, Y, Z = np.mgrid[0:volume_small.shape[0], 0:volume_small.shape[1], 0:volume_small.shape[2]]
    
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=volume_small.flatten(),
        isomin=threshold,
        isomax=volume_small.max(),
        opacity=0.1,
        surface_count=15,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        height=500
    )
    
    return fig

def visualize_slices(slices, segment_name):
    """Visualisasi 2D slices"""
    cols = st.columns(len(slices))
    for idx, (col, slice_img) in enumerate(zip(cols, slices)):
        with col:
            fig = px.imshow(slice_img, color_continuous_scale='gray')
            fig.update_layout(
                title=f"Slice {idx+1}",
                height=250,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)

# ==================== MAIN APPLICATION ====================
def main():
    # Header
    st.markdown('<h1 class="main-header">Sistem Klasifikasi ADHD</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Klasifikasi Berbasis ResNet-50 dengan Segmentasi WM/GM/CSF & Pemungutan Suara Mayoritas</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/brain.png", width=80)
        # st.title(" Configuration")
        
        st.markdown("### Model Information")
        st.info("**Architecture:** ResNet-50\n\n**Input:** sMRI 3D (DxHxW)\n\n**Output:** Binary (ADHD/TDC)")
        
        st.markdown("---")
        st.markdown("### Tahapan preposessing")
        st.markdown("""
        1. Load MRI (.nii.gz)
        2. Resize
        3. Normalisasi Min-Max
        4. Prediksi Per-tissue dengan ResNet-50
        5. Majority Voting
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Upload & Predict", " Hasil Visualisasi", "About"])
    
    with tab1:
        st.markdown("### Upload sMRI File")
        
        uploaded_file = st.file_uploader(
            "Choose a .nii.gz file",
            type=['nii', 'gz', 'nii.gz'],
            help="Upload structural MRI in NIfTI format (.nii.gz)"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.success(f"✅ File uploaded: {uploaded_file.name}")
                st.info(f"📦 File size: {uploaded_file.size / 1024 / 1024:.2f} MB")
            
            with col2:
                if st.button("Start Analysis", use_container_width=True):
                    # Validasi model files
                    model_paths = {
                        'WM': os.path.join(MODEL_DIR, 'best_medicalnet_classifier_WM.pth'),
                        'GM': os.path.join(MODEL_DIR, 'best_medicalnet_classifier_GM.pth'),
                        'CSF': os.path.join(MODEL_DIR, 'best_medicalnet_classifier_CSF.pth')
                    }
                    
                    missing_models = [k for k, v in model_paths.items() if not os.path.exists(v)]
                    if missing_models:
                        st.error(f"❌ Missing model files: {', '.join(missing_models)}")
                        return
                                   
                    try:
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Step 1: Load MRI
                        status_text.text("Loading MRI data...")
                        progress_bar.progress(10)
                        volume = load_nifti_file(uploaded_file)
                        st.session_state['volume'] = volume
                        
                        # Step 2: Segmentation
                        status_text.text("Segmenting brain tissues...")
                        progress_bar.progress(30)
                        segments = segment_tissue(volume)
                        st.session_state['segments'] = segments
                        
                        # Step 3: Load models
                        status_text.text("Loading models...")
                        progress_bar.progress(40)
                        models_dict, device = load_models(model_paths)
                        
                        # Step 4: Predictions
                        # Step 4: Predictions
                        predictions_dict = {}
                        for segment_name, model in models_dict.items():
                            status_text.text(f"Processing segment: {segment_name}...")
                            
                            vol = preprocess_volume_3d(segments[segment_name])
                            vol = vol.to(device) 

                            with torch.no_grad():
                                out = model(vol)
                                probs = torch.softmax(out, dim=1)
                                pred = probs.argmax(dim=1).item()
                                conf = probs.max(dim=1).values.item()

                            predictions_dict[segment_name] = ([pred], [conf])

                        # Step 5: Majority Voting (HANYA SATU KALI DI SINI)
                        status_text.text("Performing majority voting...")
                        progress_bar.progress(90)

                        # Pastikan menangkap 4 variabel sesuai definisi fungsi majority_voting
                        final_pred, vote_ratio, votes, avg_win_conf = majority_voting(predictions_dict)

                        if final_pred is None:
                            raise ValueError("Hasil prediksi kosong.")

                        # Simpan semua data ke session state
                        st.session_state['final_result'] = {
                            'prediction': final_pred,
                            'confidence': vote_ratio, # Ini adalah rasio kesepakatan (misal 0.66 atau 1.0)
                            'votes': votes,
                            'avg_win_conf': avg_win_conf # Ini adalah rata-rata probabilitas model yang menang
                        }
                        st.session_state['predictions'] = predictions_dict

                        # Selesaikan progress
                        progress_bar.progress(100)
                        status_text.text("✅ Analysis complete!")
                        st.balloons()
                                                
                        # --- DEBUGGING AREA ---
                        # Menampilkan isi dictionary di UI untuk verifikasi
                        with st.expander("🔍 Debug: Prediction Data"):
                            st.write(predictions_dict)
                        # ----------------------

                        st.session_state['predictions'] = predictions_dict
                        
                       
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        # Log error ke terminal juga untuk tracing lebih dalam
                        print(f"Detail Error: {e}")
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Analysis complete!")
                        
                        st.balloons()
                        
            # Display results if available
            if 'final_result' in st.session_state:
                result = st.session_state['final_result']
                label = "ADHD" if result['prediction'] == 1 else "TDC"
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Final Diagnosis</h3>
                        <h1>{label}</h1>
                        <p>&nbsp;</p> </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Tingkat Kesepakatan</h3>
                        <h1>{result['confidence']*100:.1f}%</h1>
                        <p>&nbsp;</p> </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    total_votes = result['votes'].sum()
                    avg_acc = result['avg_win_conf'] * 100
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Total Votes</h3>
                        <h1>{total_votes}</h1>
                        <p>Rata-rata Kepastian: <b>{avg_acc:.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Voting breakdown
                st.markdown("### 📊 Voting Breakdown")
                vote_data = {
                    'Class': ['TDC', 'ADHD'],
                    'Votes': [int(result['votes'][0]) if len(result['votes']) > 0 else 0, 
                             int(result['votes'][1]) if len(result['votes']) > 1 else 0]
                }
                
                fig = px.bar(vote_data, x='Class', y='Votes', 
                            color='Class',
                            color_discrete_map={'TDC': '#3498db', 'ADHD': '#e74c3c'})
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Individual segment predictions
                if 'predictions' in st.session_state:
                    st.markdown("### 🔬 Individual Segment Predictions")
                    
                    for segment_name, (preds, confs) in st.session_state['predictions'].items():
                        with st.expander(f"{segment_name} Segment Details"):
                            avg_conf = np.mean(confs) * 100
                            adhd_count = sum(preds)
                            tdc_count = len(preds) - adhd_count
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Avg Confidence", f"{avg_conf:.1f}%")
                            col2.metric("ADHD Votes", adhd_count)
                            col3.metric("TDC Votes", tdc_count)
                
    
    with tab2:
        st.markdown("## Analisis Visualisasi Citra 3D")
        
        if 'volume' in st.session_state:
            st.markdown("### 🧠 3D Brain Visualization")
            
            # --- BAGIAN LOADING ---
            with st.spinner("🏗️ Sedang merender visualisasi 3D... Ini mungkin memakan waktu beberapa detik."):
                # Karena fungsi ini di-cache, spinner hanya akan muncul di render pertama
                fig_3d = visualize_3d_brain(st.session_state['volume'])
                st.plotly_chart(fig_3d, use_container_width=True)
            
            st.markdown("---") # Garis pemisah agar lebih rapi
            st.markdown("### 📈 Volume Statistics")
            col1, col2, col3 = st.columns(3)
            
            vol = st.session_state['volume']
            col1.metric("Dimensi Volume", f"{vol.shape[0]}×{vol.shape[1]}×{vol.shape[2]}")
            col2.metric("Nilai Minimum", f"{vol.min():.2f}")
            col3.metric("Nilai Maksimum", f"{vol.max():.2f}")
            
        else:
            st.info("💡 Silakan unggah file dan jalankan analisis pada tab 'Upload & Predict' terlebih dahulu.")
    
    with tab3:
        # Header Section dengan Gaya Gradient
        st.markdown("""
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; margin-bottom: 25px;">
                <h2 style="color: white; margin: 0; text-align: center;"> Informasi Sistem & Metodologi</h2>
            </div>
        """, unsafe_allow_html=True)

        # Grid untuk Peneliti & Pembimbing
        col_dev, col_aim = st.columns([1, 1])

        with col_dev:
            st.markdown("""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #764ba2; height: 100%;">
                    <h4 style="margin-top: 0;">👤 Tim Peneliti</h4>
                    <p><b>Peneliti:</b><br>Nadhif Fajrul Minan (220411100060)</p>
                    <p><b>Pembimbing:</b><br>
                    1. Dr. Meidya Koeshardianto, S.Si., MT<br>
                    2. Dr. Cucun Very Angkoso, S.T., M.T.</p>
                </div>
            """, unsafe_allow_html=True)

        with col_aim:
            st.markdown("""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #667eea; height: 100%;">
                    <h4 style="margin-top: 0;">🎯 Tujuan Proyek</h4>
                    <p>Klasifikasi otomatis ADHD menggunakan <b>MedicalNet (3D ResNet-50)</b> untuk mendeteksi anomali struktural pada jaringan otak penderita dibandingkan kelompok kontrol (TDC).</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Metodologi dengan Expander agar rapi
        st.markdown("### 🔬 Alur Kerja Sistem (Metodologi)")
        
        m1, m2, m3 = st.columns(3)
        
        with m1:
            with st.expander("1. Segmentasi", expanded=True):
                st.write("**Tissue Segmentation**")
                st.caption("Pemisahan otomatis citra MRI menjadi White Matter (WM), Gray Matter (GM), dan CSF menggunakan thresholding intensitas.")

        with m2:
            with st.expander("2. Pra-pemrosesan", expanded=True):
                st.write("**Resampling & Norm**")
                st.caption("Standardisasi dimensi ke 64x64x64 piksel dan normalisasi intensitas Min-Max agar data seragam.")

        with m3:
            with st.expander("3. Klasifikasi", expanded=True):
                st.write("**Deep Learning**")
                st.caption("Ekstraksi fitur 3D oleh ResNet-50 untuk menghasilkan prediksi probabilitas pada tiap segmen.")

        # Petunjuk Penggunaan dengan Step Cards
        st.markdown("### 📖 Petunjuk Penggunaan")
        
        st.markdown("""
            <div style="display: flex; flex-direction: column; gap: 10px;">
                <div style="padding: 10px 20px; background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; display: flex; align-items: center;">
                    <span style="background: #764ba2; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; justify-content: center; align-items: center; margin-right: 15px; font-weight: bold;">1</span>
                    <span>Siapkan file MRI dalam format <b>.nii</b> atau <b>.nii.gz</b></span>
                </div>
                <div style="padding: 10px 20px; background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; display: flex; align-items: center;">
                    <span style="background: #764ba2; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; justify-content: center; align-items: center; margin-right: 15px; font-weight: bold;">2</span>
                    <span>Unggah file pada tab <b>'Upload & Predict'</b></span>
                </div>
                <div style="padding: 10px 20px; background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; display: flex; align-items: center;">
                    <span style="background: #764ba2; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; justify-content: center; align-items: center; margin-right: 15px; font-weight: bold;">3</span>
                    <span>Klik <b>'Start Analysis'</b> dan tunggu hingga proses voting selesai</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Penjelasan Teknis
        with st.container():
            st.markdown("### 🧠 Interpretasi Hasil")
            t1, t2 = st.columns(2)
            with t1:
                st.info("**Confidence:** Mengukur tingkat kesepakatan antar model. Jika 100%, semua jaringan (WM, GM, CSF) setuju pada satu diagnosa.")
            with t2:
                st.success("**Majority Voting:** Menentukan hasil akhir berdasarkan suara terbanyak untuk meminimalisir kesalahan prediksi individual.")

        # Footer
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.warning("⚠️ **Disklaimer:** Alat ini adalah instrumen penelitian dan bukan pengganti diagnosis medis profesional.")
        st.markdown("<p style='text-align: center; color: gray;'>Dikembangkan menggunakan Streamlit & PyTorch</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()