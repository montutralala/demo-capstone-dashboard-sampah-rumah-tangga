"""
Dashboard Interaktif Trashkara — Analisis Data Sampah Rumah Tangga
CC26-PSU013 | Data Science
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr, mannwhitneyu
import re
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Trashkara Dashboard",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f4c35 0%, #1a7a52 50%, #2ecc71 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .main-header p  { font-size: 1rem; margin: 0.4rem 0 0; opacity: 0.85; }

    .metric-card {
        background: white;
        border: 1px solid #e8f5e9;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-card .label { font-size: 0.78rem; color: #666; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .value { font-size: 1.9rem; font-weight: 700; color: #0f4c35; margin: 0.2rem 0; }
    .metric-card .sub   { font-size: 0.82rem; color: #888; }

    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #0f4c35;
        border-left: 4px solid #2ecc71;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem;
    }

    .bq-box {
        background: #f0fdf4;
        border: 1.5px solid #86efac;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        margin-bottom: 1rem;
        font-size: 0.9rem;
        color: #14532d;
        font-style: italic;
    }

    .insight-box {
        background: #fffbeb;
        border: 1.5px solid #fde68a;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        margin-top: 1rem;
        font-size: 0.88rem;
        color: #78350f;
    }
    .insight-box strong { color: #92400e; }

    .stat-result {
        background: #f0f9ff;
        border: 1.5px solid #7dd3fc;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        font-size: 0.88rem;
        color: #0c4a6e;
    }

    [data-testid="stSidebar"] { background: #0f4c35; }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label { color: #d1fae5 !important; }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] *,
    [data-testid="stSidebar"] .stSelectbox input,
    [data-testid="stSidebar"] .stSelectbox span {
        color: #111111 !important;
    }
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="popover"] li,
    [data-baseweb="menu"] * { 
        color: #111111 !important; 
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #f0fdf4;
        border-radius: 8px 8px 0 0;
        border: 1px solid #d1fae5;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        color: #0f4c35;
    }
    .stTabs [aria-selected="true"] {
        background: #0f4c35 !important;
        color: white !important;
        border-color: #0f4c35 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING & PROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))

    # Coba model_ready dulu (sudah ada semua fitur)
    paths = [
        os.path.join(base, "data/dataset_trashkara_model_ready.csv"),
        os.path.join(base, "data/dataset_trashkara_processed.csv"),
        os.path.join(base, "data/main_data.csv"),
    ]
    df = None
    for p in paths:
        try:
            df = pd.read_csv(p)
            break
        except FileNotFoundError:
            continue

    if df is None:
        # fallback ke clean dan proses manual
        df_raw = pd.read_csv(os.path.join(base, "data/dataset_trashkara_clean.csv"))
        df = df_raw.copy()

        for col in ['Dapat Terurai', 'Daur Ulang', 'Nilai Jual']:
            if df[col].dtype == object:
                df[col] = df[col].map({'Ya': 1, 'Tidak': 0})

        def parse_waktu_urai_to_bulan(val):
            if pd.isnull(val): return np.nan
            v = str(val).lower().replace('\u2013', '-')
            nums = re.findall(r'[\d\.]+', v)
            if not nums: return np.nan
            mid = np.mean([float(n) for n in nums])
            if 'juta' in v and 'tahun' in v: return mid * 1_000_000 * 12
            elif 'tahun' in v: return mid * 12
            elif 'bulan' in v: return mid
            elif 'minggu' in v: return mid / 4.33
            return np.nan

        df['waktu_urai_bulan'] = df['Waktu Urai'].apply(parse_waktu_urai_to_bulan)
        df['kategori_urai'] = df['waktu_urai_bulan'].apply(
            lambda b: 'Cepat' if pd.notnull(b) and b <= 6
                      else ('Sedang' if pd.notnull(b) and b <= 600 else 'Lama'))

        kes_map = {'Mudah': 1, 'Sedang': 2, 'Sulit': 3, 'Sangat Sulit': 4}
        df['kesulitan_encoded'] = df['Kesulitan Daur Ulang'].map(kes_map)

        nilai_jual_map = {
            'Rendah (< Rp 500/kg)': 1, 'Sedang (Rp 500-2.000/kg)': 2,
            'Sedang (Rp 2.000\u20134.000/kg)': 3, 'Tinggi (> Rp 2.000/kg)': 4,
            'Tinggi (> Rp 5.000/kg)': 5
        }
        biaya_map = {
            'Rendah (< Rp 500/kg)': 1, 'Sedang (Rp 500-2.000/kg)': 2,
            'Sedang (Rp 1.500\u20133.000/kg)': 3, 'Tinggi (> Rp 2.000/kg)': 4,
            'Sangat Tinggi (> Rp 5.000/kg)': 5
        }
        df['nilai_jual_encoded'] = df['Nilai Jual (Rp/kg)'].map(nilai_jual_map)
        df['biaya_proses_encoded'] = df['Biaya Proses (Rp/kg)'].map(biaya_map)
        df['beban_biaya_score'] = df['nilai_jual_encoded'] + df['biaya_proses_encoded']

        def tentukan_struktur(row):
            if row['biaya_proses_encoded'] > row['nilai_jual_encoded']:
                return 'Biaya Operasional Tinggi'
            elif row['biaya_proses_encoded'] == row['nilai_jual_encoded']:
                return 'Impas'
            else:
                return 'Ekonomis'

        df['Struktur_Biaya'] = df.apply(tentukan_struktur, axis=1)
        df['cost_burden'] = df['biaya_proses_encoded'] - df['nilai_jual_encoded']
        df['high_cost_burden'] = (df['cost_burden'] >= 2).astype(int)

        emisi_n = (df['Emisi CO2e (kg)'] - df['Emisi CO2e (kg)'].min()) / \
                  (df['Emisi CO2e (kg)'].max() - df['Emisi CO2e (kg)'].min())
        urai_n  = np.log1p(df['waktu_urai_bulan'].fillna(0))
        urai_n  = (urai_n - urai_n.min()) / (urai_n.max() - urai_n.min())

        df['env_impact_score']    = emisi_n * 0.4 + urai_n * 0.4 + (1 - df['Dapat Terurai'].fillna(0)) * 0.2
        df['econ_burden_score']   = (df['biaya_proses_encoded'].fillna(2)/5)*0.6 + (1 - df['nilai_jual_encoded'].fillna(2)/5)*0.4
        df['recyclability_index'] = (df['Daur Ulang'].fillna(0)*0.4 +
                                      (df['nilai_jual_encoded'].fillna(1)/5)*0.35 +
                                      (1 - df['kesulitan_encoded'].fillna(2)/5)*0.25)

    # Pastikan cost_burden ada
    if 'cost_burden' not in df.columns:
        df['cost_burden'] = df['biaya_proses_encoded'] - df['nilai_jual_encoded']
    if 'high_cost_burden' not in df.columns:
        df['high_cost_burden'] = (df['cost_burden'] >= 2).astype(int)

    # Normalisasi label Struktur_Biaya (bisa mengandung newline dari model_ready)
    if 'Struktur_Biaya' in df.columns:
        mapping = {}
        for v in df['Struktur_Biaya'].unique():
            v_str = str(v)
            if 'Ekonomis' in v_str or 'Menguntungkan' in v_str:
                mapping[v] = 'Ekonomis'
            elif 'Impas' in v_str:
                mapping[v] = 'Impas'
            else:
                mapping[v] = 'Biaya Operasional Tinggi'
        df['Struktur_Biaya'] = df['Struktur_Biaya'].map(mapping)
    else:
        def tentukan_struktur(row):
            if row['biaya_proses_encoded'] > row['nilai_jual_encoded']:
                return 'Biaya Operasional Tinggi'
            elif row['biaya_proses_encoded'] == row['nilai_jual_encoded']:
                return 'Impas'
            else:
                return 'Ekonomis'
        df['Struktur_Biaya'] = df.apply(tentukan_struktur, axis=1)

    return df

df = load_data()

CAT_COLOR = {'Anorganik': '#3498db', 'B3': '#e74c3c', 'Organik': '#2ecc71'}
KES_ORDER  = ['Mudah', 'Sedang', 'Sulit', 'Sangat Sulit']
CATLIST    = ['Anorganik', 'B3', 'Organik']

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/recycle-sign.png", width=80)
    st.title("♻️ Trashkara")
    st.markdown("**Dataset Klasifikasi Sampah Indonesia**")
    st.markdown("---")

    page = st.selectbox("📌 Navigasi", [
        "🏠 Overview",
        "🌿 Q1 — Emisi CO₂e",
        "💰 Q2 — Beban Biaya",
        "🔬 Feature Engineering",
        "🧪 A/B Testing",
    ])

    st.markdown("---")
    st.subheader("🔍 Filter Data")

    kat_filter = st.multiselect(
        "Kategori Sampah",
        options=CATLIST,
        default=CATLIST
    )

    kesulitan_filter = st.multiselect(
        "Tingkat Kesulitan Daur Ulang",
        options=KES_ORDER,
        default=KES_ORDER
    )

    metode_filter = st.multiselect(
        "Metode Pengumpulan",
        options=df['Metode Pengumpulan'].unique().tolist(),
        default=df['Metode Pengumpulan'].unique().tolist()
    )

    emisi_range = st.slider(
        "Rentang Emisi CO₂e (kg)",
        float(df['Emisi CO2e (kg)'].min()),
        float(df['Emisi CO2e (kg)'].max()),
        (float(df['Emisi CO2e (kg)'].min()), float(df['Emisi CO2e (kg)'].max())),
        step=0.1
    )

    st.markdown("---")
    st.markdown("### 📁 Dataset Info")
    st.info(f"**Total Baris:** {len(df):,}\n\n**Jenis Sampah:** {df['Nama Sampah'].nunique()}\n\n**Kolom:** {df.shape[1]}")

# Apply filters
df_f = df[
    df['Kategori'].isin(kat_filter) &
    df['Kesulitan Daur Ulang'].isin(kesulitan_filter) &
    df['Metode Pengumpulan'].isin(metode_filter) &
    df['Emisi CO2e (kg)'].between(emisi_range[0], emisi_range[1])
]

# ════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class="main-header">
        <h1>♻️ Trashkara — Dashboard Analisis Data Sampah</h1>
        <p>Intelligent Waste Classifier & Generative Upcycling Assistant | CC26-PSU013 | Data Science 2026</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_data = [
        (c1, "Total Sampel",      f"{len(df_f):,}",                         "baris tersaring"),
        (c2, "Jenis Sampah",      f"{df_f['Nama Sampah'].nunique()}",        "dari 47 total"),
        (c3, "Rata-rata Emisi",   f"{df_f['Emisi CO2e (kg)'].mean():.3f} kg","CO₂e per sampel"),
        (c4, "Avg Cost Burden",   f"{df_f['cost_burden'].mean():.2f}",       "BP − NJ (encoded)"),
        (c5, "High-Cost Burden",  f"{df_f['high_cost_burden'].mean()*100:.0f}%", "sampel"),
    ]
    for col, label, val, sub in kpi_data:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{val}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Gambaran Umum Dataset Trashkara 2026</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Komposisi Kategori
        cat_j = df_f.groupby('Kategori')['Nama Sampah'].nunique().reset_index()
        cat_j.columns = ['Kategori', 'Jumlah Jenis']
        fig_pie = px.pie(
            cat_j, names='Kategori', values='Jumlah Jenis',
            color='Kategori', color_discrete_map=CAT_COLOR,
            title='Komposisi Jenis Sampah per Kategori',
            hole=0.4
        )
        fig_pie.update_traces(textposition='outside', textinfo='percent+label')
        fig_pie.update_layout(height=380, showlegend=True, title_font_size=14)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Kesulitan Daur Ulang
        kes_cnt = df_f['Kesulitan Daur Ulang'].value_counts().reindex(KES_ORDER).reset_index()
        kes_cnt.columns = ['Kesulitan', 'Jumlah']
        fig_bar = px.bar(
            kes_cnt, x='Kesulitan', y='Jumlah',
            color='Kesulitan',
            color_discrete_sequence=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'],
            title='Distribusi Tingkat Kesulitan Daur Ulang',
            text='Jumlah'
        )
        fig_bar.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_bar.update_layout(height=380, showlegend=False, title_font_size=14,
                               xaxis_title='', yaxis_title='Frekuensi')
        st.plotly_chart(fig_bar, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Distribusi Emisi CO2e
        fig_hist = px.histogram(
            df_f, x='Emisi CO2e (kg)', nbins=30,
            color='Kategori', color_discrete_map=CAT_COLOR,
            title='Distribusi Emisi CO₂e per Kategori',
            barmode='overlay', opacity=0.7
        )
        mean_val = df_f['Emisi CO2e (kg)'].mean()
        fig_hist.add_vline(x=mean_val, line_dash='dash', line_color='#7f0000',
                           annotation_text=f'Mean = {mean_val:.2f} kg', annotation_position='top right')
        fig_hist.update_layout(height=360, title_font_size=14,
                                xaxis_title='Emisi CO₂e (kg)', yaxis_title='Frekuensi')
        st.plotly_chart(fig_hist, use_container_width=True)

    with col4:
        # Kategori Waktu Urai
        urai_cnt = df_f['kategori_urai'].value_counts().reset_index()
        urai_cnt.columns = ['Kategori Urai', 'Jumlah']
        fig_urai = px.bar(
            urai_cnt, x='Kategori Urai', y='Jumlah',
            color='Kategori Urai',
            color_discrete_sequence=['#2ecc71', '#f1c40f', '#e74c3c'],
            title='Distribusi Kategori Waktu Urai Material',
            text='Jumlah'
        )
        fig_urai.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_urai.update_layout(height=360, showlegend=False, title_font_size=14,
                                xaxis_title='', yaxis_title='Frekuensi')
        st.plotly_chart(fig_urai, use_container_width=True)

    st.markdown('<div class="section-header">Statistik Deskriptif per Kategori</div>', unsafe_allow_html=True)
    stats = df_f.groupby('Kategori').agg(
        Jumlah_Data=('Nama Sampah', 'count'),
        Jenis_Sampah=('Nama Sampah', 'nunique'),
        Mean_Emisi=('Emisi CO2e (kg)', 'mean'),
        Median_Emisi=('Emisi CO2e (kg)', 'median'),
        Mean_Beban_Biaya=('beban_biaya_score', 'mean'),
        Mean_Daur_Ulang=('Daur Ulang', 'mean'),
    ).round(3).reset_index()
    stats.columns = ['Kategori', 'Jumlah Data', 'Jenis Unik', 'Mean Emisi (kg)',
                     'Median Emisi (kg)', 'Mean Beban Biaya', 'Proporsi Bisa Daur Ulang']
    st.dataframe(stats, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════
# PAGE: Q1 — EMISI CO2E
# ════════════════════════════════════════════════════════
elif page == "🌿 Q1 — Emisi CO₂e":
    st.markdown("""
    <div class="main-header">
        <h1>🌿 Q1 — Emisi CO₂e & Kesulitan Daur Ulang</h1>
        <p>Analisis distribusi emisi dan korelasi dengan tingkat kesulitan daur ulang</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="bq-box">
    ❓ <strong>Pertanyaan Bisnis 1:</strong> Kategori sampah manakah yang menyumbang rata-rata emisi CO₂e tertinggi,
    dan apakah tingkat kesulitan daur ulang berkorelasi positif secara signifikan dengan besarnya emisi yang dihasilkan?
    <br><em>(Diuji dengan Spearman Correlation, α=0.05)</em>
    </div>""", unsafe_allow_html=True)

    # KPI per kategori
    c1, c2, c3 = st.columns(3)
    for i, cat in enumerate(CATLIST):
        if cat in df_f['Kategori'].unique():
            avg = df_f[df_f['Kategori'] == cat]['Emisi CO2e (kg)'].mean()
            [c1, c2, c3][i].metric(f"{cat} — Avg Emisi", f"{avg:.3f} kg")

    col1, col2 = st.columns(2)

    with col1:
        fig_box = px.box(
            df_f, x='Kategori', y='Emisi CO2e (kg)',
            color='Kategori', color_discrete_map=CAT_COLOR,
            title='Distribusi Emisi CO₂e per Kategori (Box Plot)',
            points='outliers'
        )
        fig_box.update_layout(height=420, showlegend=False, title_font_size=14)
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption("💡 Organik memiliki median emisi tertinggi. Dekomposisi anaerobik melepas CH₄ (25× lebih kuat dari CO₂).")

    with col2:
        emisi_kes = df_f.groupby('Kesulitan Daur Ulang')['Emisi CO2e (kg)'].agg(
            Mean='mean', SE=lambda x: x.sem()).reindex(KES_ORDER).dropna().reset_index()
        emisi_kes.columns = ['Kesulitan', 'Mean Emisi', 'SE']
        fig_kes = px.bar(
            emisi_kes, x='Kesulitan', y='Mean Emisi',
            color='Kesulitan',
            color_discrete_sequence=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'],
            error_y='SE', title='Rata-rata Emisi CO₂e per Tingkat Kesulitan (±SE)',
            text='Mean Emisi'
        )
        fig_kes.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_kes.update_layout(height=420, showlegend=False, title_font_size=14,
                               xaxis_title='Tingkat Kesulitan', yaxis_title='Avg Emisi CO₂e (kg)')
        st.plotly_chart(fig_kes, use_container_width=True)
        st.caption("💡 Tren naik Mudah → Sangat Sulit. Error bar (SE) kecil → estimasi stabil.")

    col3, col4 = st.columns(2)

    with col3:
        emisi_cat = df_f.groupby('Kategori')['Emisi CO2e (kg)'].mean().reset_index().sort_values('Emisi CO2e (kg)', ascending=False)
        fig_emisi = px.bar(
            emisi_cat, x='Kategori', y='Emisi CO2e (kg)',
            color='Kategori', color_discrete_map=CAT_COLOR,
            title='Rata-rata Emisi CO₂e per Kategori',
            text='Emisi CO2e (kg)'
        )
        fig_emisi.update_traces(texttemplate='%{text:.3f} kg', textposition='outside')
        fig_emisi.update_layout(height=400, showlegend=False, title_font_size=14,
                                 yaxis_title='Rata-rata Emisi CO₂e (kg)', xaxis_title='')
        st.plotly_chart(fig_emisi, use_container_width=True)

    with col4:
        fig_violin = px.violin(
            df_f, x='Kesulitan Daur Ulang', y='Emisi CO2e (kg)',
            color='Kesulitan Daur Ulang',
            category_orders={'Kesulitan Daur Ulang': KES_ORDER},
            title='Distribusi Emisi per Tingkat Kesulitan (Violin)',
            box=True, points=False
        )
        fig_violin.update_layout(height=400, showlegend=False, title_font_size=14)
        st.plotly_chart(fig_violin, use_container_width=True)

    # Uji Spearman
    st.markdown('<div class="section-header">📐 Uji Spearman: Korelasi Kesulitan ↔ Emisi CO₂e</div>', unsafe_allow_html=True)
    df_corr = df_f[['kesulitan_encoded', 'Emisi CO2e (kg)']].dropna()
    rho, pval = spearmanr(df_corr['kesulitan_encoded'], df_corr['Emisi CO2e (kg)'])

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Spearman ρ (rho)", f"{rho:.4f}", delta="Korelasi Positif" if rho > 0 else "Korelasi Negatif")
    mc2.metric("P-Value", f"{pval:.6f}", delta="Signifikan ✅" if pval < 0.05 else "Tidak Signifikan ❌")
    mc3.metric("Kesimpulan H₀", "TOLAK H₀" if pval < 0.05 else "GAGAL TOLAK H₀",
               delta="p < 0.05" if pval < 0.05 else "p ≥ 0.05")

    if pval < 0.05:
        st.success(f"✅ Korelasi positif **signifikan** — p={pval:.6f} < 0.05. Semakin sulit didaur ulang, emisi semakin besar (ρ={rho:.4f}).")
        st.info("💡 **Interpretasi:** Meski kekuatan korelasi lemah (ρ≈0.13), dengan n besar ukuran efek kecil pun bermakna secara statistik. Banyak faktor lain (komposisi kimia, proses produksi) turut mempengaruhi emisi.")

    # Top 15 emisi tertinggi
    st.markdown('<div class="section-header">🏆 Top 15 Jenis Sampah — Emisi CO₂e Tertinggi</div>', unsafe_allow_html=True)
    top_emisi = df_f.groupby(['Nama Sampah', 'Kategori'])['Emisi CO2e (kg)'].mean().reset_index()
    top_emisi = top_emisi.sort_values('Emisi CO2e (kg)', ascending=False).head(15)
    fig_top = px.bar(
        top_emisi, y='Nama Sampah', x='Emisi CO2e (kg)',
        color='Kategori', color_discrete_map=CAT_COLOR,
        orientation='h', title='Top 15 Jenis Sampah dengan Rata-rata Emisi CO₂e Tertinggi',
        text='Emisi CO2e (kg)'
    )
    fig_top.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_top.update_layout(height=520, title_font_size=14, yaxis={'categoryorder': 'total ascending'},
                           xaxis_title='Rata-rata Emisi CO₂e (kg)', yaxis_title='')
    st.plotly_chart(fig_top, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <strong>💡 Kesimpulan Q1:</strong><br>
    • Kategori <strong>Organik</strong> menghasilkan emisi CO₂e rata-rata tertinggi akibat proses dekomposisi yang melepas gas metana (CH₄).<br>
    • Uji Spearman menunjukkan korelasi positif <strong>signifikan namun lemah</strong> (ρ ≈ 0.13, p &lt; 0.05).<br>
    • <strong>Karakteristik bahan dasar</strong> sampah lebih menentukan emisi daripada kesulitan daur ulang.<br>
    • Prioritaskan infrastruktur <em>composting & biogas</em> untuk menekan emisi GRK dari sampah organik.
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# PAGE: Q2 — BEBAN BIAYA
# ════════════════════════════════════════════════════════
elif page == "💰 Q2 — Beban Biaya":
    st.markdown("""
    <div class="main-header">
        <h1>💰 Q2 — Struktur Beban Biaya Operasional</h1>
        <p>Analisis distribusi beban biaya lintas kategori dan identifikasi high-cost burden</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="bq-box">
    ❓ <strong>Pertanyaan Bisnis 2:</strong> Bagaimana distribusi struktur beban biaya (nilai jual mentah dan biaya proses)
    lintas kategori utama, dan jenis sampah mana yang masuk kuadran high-cost burden?
    <br><br>
    📌 <strong>Nilai Jual (Rp/kg)</strong> = harga sampah mentah yang dijual ke pabrik/pengepul sebelum diolah.<br>
    📌 <strong>Biaya Proses (Rp/kg)</strong> = biaya operasional industri: mesin, energi, tenaga kerja, dll.<br>
    📌 <strong>Cost Burden</strong> = Biaya Proses − Nilai Jual → semakin positif = defisit industri semakin besar.
    </div>""", unsafe_allow_html=True)

    # KPI per kategori
    c1, c2, c3 = st.columns(3)
    for i, cat in enumerate(CATLIST):
        if cat in df_f['Kategori'].unique():
            avg_cb  = df_f[df_f['Kategori'] == cat]['cost_burden'].mean()
            pct_hcb = df_f[df_f['Kategori'] == cat]['high_cost_burden'].mean() * 100
            [c1, c2, c3][i].metric(f"{cat}", f"Avg CB: {avg_cb:.2f}", f"{pct_hcb:.0f}% high-cost")

    col1, col2 = st.columns(2)

    with col1:
        # Stacked bar distribusi struktur biaya
        komposisi = pd.crosstab(df_f['Kategori'], df_f['Struktur_Biaya'], normalize='index') * 100
        for col_name in ['Ekonomis', 'Impas', 'Biaya Operasional Tinggi']:
            if col_name not in komposisi.columns:
                komposisi[col_name] = 0
        komposisi = komposisi[['Ekonomis', 'Impas', 'Biaya Operasional Tinggi']].reset_index()

        fig_stacked = go.Figure()
        colors_stack = {'Ekonomis': '#2ecc71', 'Impas': '#f1c40f', 'Biaya Operasional Tinggi': '#e74c3c'}
        for status, color in colors_stack.items():
            fig_stacked.add_trace(go.Bar(
                name=status, x=komposisi['Kategori'], y=komposisi[status],
                marker_color=color, text=[f"{v:.1f}%" for v in komposisi[status]],
                textposition='inside', textfont=dict(color='white', size=11)
            ))
        fig_stacked.update_layout(
            barmode='stack', title='Distribusi Struktur Beban Biaya per Kategori Utama',
            height=420, title_font_size=14, yaxis_title='Persentase (%)',
            legend=dict(orientation='h', yanchor='bottom', y=-0.3)
        )
        st.plotly_chart(fig_stacked, use_container_width=True)
        st.caption("💡 B3: 83% sampel berada pada Biaya Operasional Tinggi. Organik: 0% beban tinggi — paling viable secara ekonomi.")

    with col2:
        # Violin plot cost_burden
        kats_v = [k for k in CATLIST if k in df_f['Kategori'].unique()]
        fig_violin = go.Figure()
        for cat in kats_v:
            fig_violin.add_trace(go.Violin(
                y=df_f[df_f['Kategori'] == cat]['cost_burden'],
                name=cat, box_visible=True, meanline_visible=True,
                fillcolor=CAT_COLOR[cat], opacity=0.7,
                line_color=CAT_COLOR[cat]
            ))
        fig_violin.add_hline(y=0, line_dash='dash', line_color='black', opacity=0.6,
                              annotation_text='Break-even (0)', annotation_position='right')
        fig_violin.add_hline(y=2, line_dash='dot', line_color='red', opacity=0.6,
                              annotation_text='High-cost threshold (2)', annotation_position='right')
        fig_violin.update_layout(
            title='Distribusi Cost Burden per Kategori (Violin Plot)',
            height=420, title_font_size=14, yaxis_title='Cost Burden', showlegend=True
        )
        st.plotly_chart(fig_violin, use_container_width=True)
        st.caption("💡 B3 memusat di atas threshold 2. Organik memusat di bawah nol. Anorganik paling tersebar.")

    col3, col4 = st.columns(2)

    with col3:
        ekon = df_f.groupby('Kategori')[['nilai_jual_encoded', 'biaya_proses_encoded']].mean().reset_index()
        fig_grouped = go.Figure()
        fig_grouped.add_trace(go.Bar(
            name='Nilai Jual Mentah', x=ekon['Kategori'], y=ekon['nilai_jual_encoded'],
            marker_color='#2c3e50', text=ekon['nilai_jual_encoded'].round(2), textposition='outside'
        ))
        fig_grouped.add_trace(go.Bar(
            name='Biaya Proses Industri', x=ekon['Kategori'], y=ekon['biaya_proses_encoded'],
            marker_color='#e67e22', text=ekon['biaya_proses_encoded'].round(2), textposition='outside'
        ))
        fig_grouped.update_layout(
            barmode='group', title='Nilai Jual vs Biaya Proses per Kategori (Skala 1–5)',
            height=380, title_font_size=14, yaxis_title='Encoded Value (1–5)',
            legend=dict(orientation='h', yanchor='bottom', y=-0.3)
        )
        st.plotly_chart(fig_grouped, use_container_width=True)

    with col4:
        bb_cat = df_f.groupby('Kategori')['cost_burden'].mean().reset_index().sort_values('cost_burden')
        fig_hbar = px.bar(
            bb_cat, y='Kategori', x='cost_burden',
            color='Kategori', color_discrete_map=CAT_COLOR,
            orientation='h', title='Rata-rata Cost Burden per Kategori',
            text='cost_burden'
        )
        fig_hbar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_hbar.add_vline(x=0, line_dash='dash', line_color='black', opacity=0.5)
        fig_hbar.update_layout(height=380, showlegend=False, title_font_size=14,
                                xaxis_title='Cost Burden', yaxis_title='')
        st.plotly_chart(fig_hbar, use_container_width=True)

    # Scatter kuadran
    st.markdown('<div class="section-header">🎯 Kuadran: Nilai Jual Mentah vs Biaya Proses</div>', unsafe_allow_html=True)
    jenis_agg = df_f.groupby(['Nama Sampah', 'Kategori']).agg(
        nj=('nilai_jual_encoded', 'mean'),
        bp=('biaya_proses_encoded', 'mean'),
        cb=('cost_burden', 'mean'),
        emisi=('Emisi CO2e (kg)', 'mean')
    ).reset_index()
    jenis_agg['high_cost'] = jenis_agg['cb'] >= 2

    fig_scatter = px.scatter(
        jenis_agg, x='nj', y='bp',
        color='Kategori', color_discrete_map=CAT_COLOR,
        size='emisi', size_max=30,
        hover_name='Nama Sampah',
        hover_data={'nj': ':.2f', 'bp': ':.2f', 'cb': ':.2f'},
        title='Kuadran: Nilai Jual Mentah vs Biaya Proses<br><sup>Ukuran titik = emisi CO₂e | Label merah = high-cost burden</sup>',
        labels={'nj': 'Nilai Jual Mentah (Encoded 1–5)', 'bp': 'Biaya Proses Industri (Encoded 1–5)'}
    )
    # Break-even line
    x_line = np.linspace(0.8, 5.2, 100)
    fig_scatter.add_trace(go.Scatter(x=x_line, y=x_line, mode='lines', name='Break-even (BP=NJ)',
                                      line=dict(dash='dash', color='black', width=1), opacity=0.5))
    fig_scatter.add_trace(go.Scatter(x=x_line, y=x_line + 2, mode='lines', name='High-cost threshold (CB≥2)',
                                      line=dict(dash='dot', color='red', width=1.5), opacity=0.7))
    # Label high-cost items
    for _, row in jenis_agg[jenis_agg['high_cost']].iterrows():
        fig_scatter.add_annotation(
            x=row['nj'], y=row['bp'], text=row['Nama Sampah'],
            showarrow=True, arrowhead=2, arrowcolor='#c0392b',
            font=dict(color='#7f0000', size=10), bgcolor='rgba(255,255,255,0.8)'
        )
    fig_scatter.update_layout(height=520, title_font_size=14)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Tabel high-cost burden
    hcb_list = jenis_agg[jenis_agg['high_cost']].sort_values('cb', ascending=False)
    st.subheader(f"⚠️ Jenis Sampah High-Cost Burden ({len(hcb_list)} jenis)")
    st.dataframe(hcb_list[['Nama Sampah', 'Kategori', 'nj', 'bp', 'cb']].rename(
        columns={'nj': 'NJ Encoded', 'bp': 'BP Encoded', 'cb': 'Cost Burden'}),
        use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <strong>💡 Kesimpulan Q2:</strong><br>
    • Kategori <strong>B3 memiliki beban biaya tertinggi</strong> — 83% sampel masuk Biaya Operasional Tinggi.<br>
    • <strong>Kabel listrik bekas</strong> (CB=0) dan <strong>Aerosol</strong> (CB=2) masuk zona high-cost burden.<br>
    • <strong>Organik</strong> paling viable: semua sampel memiliki nilai jual ≥ biaya proses.<br>
    • Rekomendasi: Terapkan kebijakan <em>Extended Producer Responsibility (EPR)</em> untuk mengurangi beban industri.
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# PAGE: FEATURE ENGINEERING
# ════════════════════════════════════════════════════════
elif page == "🔬 Feature Engineering":
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Feature Engineering</h1>
        <p>Fitur komposit untuk pemodelan Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    fe_cols = ['env_impact_score', 'econ_burden_score', 'recyclability_index']
    fe_avail = [c for c in fe_cols if c in df_f.columns]

    if fe_avail:
        cols_fe = st.columns(3)
        labels_fe = {
            'env_impact_score': ('⚠️ Avg Env Impact Score', 'Dampak lingkungan (emisi+urai+terurai)'),
            'econ_burden_score': ('💸 Avg Econ Burden Score', 'Beban biaya ekonomi industri'),
            'recyclability_index': ('♻️ Avg Recyclability Index', 'Kemudahan & potensi daur ulang'),
        }
        for i, feat in enumerate(fe_avail):
            lbl, hlp = labels_fe[feat]
            cols_fe[i].metric(lbl, f"{df_f[feat].mean():.3f}", help=hlp)

        col1, col2 = st.columns(2)

        with col1:
            fe_stats = df_f.groupby('Kategori')[fe_avail].mean().round(3).reset_index()
            fig_radar = go.Figure()
            categories_radar = fe_avail
            for _, row in fe_stats.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row[f] for f in fe_avail],
                    theta=categories_radar,
                    fill='toself',
                    name=row['Kategori'],
                    line_color=CAT_COLOR.get(row['Kategori'], '#888'),
                    opacity=0.7
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title='Radar Chart: Profil Skor Komposit per Kategori',
                height=440, showlegend=True, title_font_size=14
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col2:
            st.markdown("**Rata-rata Skor Fitur Baru per Kategori**")
            fe_display = fe_stats.copy()
            fe_display.columns = ['Kategori'] + [c.replace('_', ' ').title() for c in fe_avail]
            st.dataframe(fe_display, use_container_width=True, hide_index=True)

            st.markdown("""
            <div style="font-size:0.85rem; color:#555; margin-top:1rem; padding:0.8rem; background:#f9fafb; border-radius:8px;">
            <strong>Keterangan Skala:</strong><br>
            • <em>Env Impact Score</em>: 0 = Aman, 1 = Sangat Merusak Lingkungan<br>
            • <em>Econ Burden Score</em>: 0 = Menguntungkan, 1 = Beban Ekonomi Berat<br>
            • <em>Recyclability Index</em>: 0 = Sangat Sulit, 1 = Sangat Mudah Didaur Ulang
            </div>""", unsafe_allow_html=True)

        # Bubble Chart
        st.markdown('<div class="section-header">Bubble Chart: Dampak Lingkungan vs Beban Ekonomi</div>', unsafe_allow_html=True)
        fe_agg = df_f.groupby(['Nama Sampah', 'Kategori']).agg(
            env=('env_impact_score', 'mean'),
            econ=('econ_burden_score', 'mean'),
            ri=('recyclability_index', 'mean')
        ).reset_index()

        fig_bubble = px.scatter(
            fe_agg, x='econ', y='env',
            size='ri', size_max=30,
            color='Kategori', color_discrete_map=CAT_COLOR,
            hover_name='Nama Sampah',
            hover_data={'econ': ':.3f', 'env': ':.3f', 'ri': ':.3f'},
            title='Bubble Chart: Dampak Lingkungan vs Beban Ekonomi<br><sup>Ukuran bubble = Recyclability Index (makin besar = makin mudah didaur ulang)</sup>',
            labels={'econ': 'Economic Burden Score (0=Ringan, 1=Berat)',
                    'env': 'Environmental Impact Score (0=Aman, 1=Merusak)'}
        )
        fig_bubble.add_hline(y=fe_agg['env'].mean(), line_dash='dash', line_color='gray', opacity=0.5)
        fig_bubble.add_vline(x=fe_agg['econ'].mean(), line_dash='dot', line_color='gray', opacity=0.5)
        fig_bubble.add_annotation(x=0.85, y=0.85, text="🔴 Kuadran Kritis", showarrow=False,
                                   font=dict(color='#e74c3c', size=11))
        fig_bubble.add_annotation(x=0.2, y=0.2, text="🟢 Kuadran Emas", showarrow=False,
                                   font=dict(color='#27ae60', size=11))
        fig_bubble.update_layout(height=540, title_font_size=14)
        st.plotly_chart(fig_bubble, use_container_width=True)

        # Distribusi per fitur
        feat_sel = st.selectbox("Lihat distribusi fitur:", fe_avail)
        fig_dist = px.histogram(
            df_f, x=feat_sel, color='Kategori', color_discrete_map=CAT_COLOR,
            barmode='overlay', opacity=0.7, nbins=20,
            title=f'Distribusi {feat_sel} per Kategori'
        )
        fig_dist.update_layout(height=380, title_font_size=14)
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>💡 Insight Feature Engineering:</strong><br>
        • <strong>B3 (Merah)</strong> — paling kritis: beban ekonomi tertinggi dan dampak lingkungan tertinggi.<br>
        • <strong>Organik (Hijau)</strong> — paling aman: beban biaya termurah dan recyclability tertinggi.<br>
        • <strong>Anorganik (Biru)</strong> — posisi menengah, sebaran luas karena variasi bahan yang beragam.
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("Fitur komposit tidak ditemukan di dataset. Pastikan menggunakan dataset_trashkara_model_ready.csv.")

# ════════════════════════════════════════════════════════
# PAGE: A/B TESTING
# ════════════════════════════════════════════════════════
elif page == "🧪 A/B Testing":
    st.markdown("""
    <div class="main-header">
        <h1>🧪 A/B Testing — Efektivitas Metode Pengumpulan</h1>
        <p>Uji Mann-Whitney U: metode terstruktur vs konvensional</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="bq-box">
    🧪 <strong>Hipotesis A/B Testing:</strong><br>
    H₁: Metode pengumpulan terstruktur (Bank Sampah / TPS 3R) menghasilkan cost_burden yang <em>lebih rendah secara signifikan</em>
    dibandingkan metode konvensional (Pengepul / Drop Box / Pickup Door-to-Door).<br>
    <em>Uji: Mann-Whitney U | One-Tailed | α = 0.05</em>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
| Parameter | Detail |
|---|---|
| **H₀** | Tidak ada perbedaan cost_burden antara metode terstruktur dan konvensional |
| **H₁** | Metode terstruktur (Bank Sampah & TPS 3R) menghasilkan cost_burden **lebih rendah** |
| **Uji** | Mann-Whitney U (non-parametric) |
| **α** | 0.05 |
    """)

    gA = df_f[df_f['Metode Pengumpulan'].isin(['Bank Sampah', 'TPS 3R'])]['cost_burden'].dropna()
    gB = df_f[df_f['Metode Pengumpulan'].isin(['Pengepul', 'Drop Box', 'Pickup Door-to-Door'])]['cost_burden'].dropna()

    if len(gA) > 0 and len(gB) > 0:
        stat_ab, pval_ab = mannwhitneyu(gA, gB, alternative='less')

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Group A (n)", f"{len(gA):,}", delta="Terstruktur")
        c2.metric("Mean Cost Burden A", f"{gA.mean():.4f}")
        c3.metric("Group B (n)", f"{len(gB):,}", delta="Konvensional")
        c4.metric("Mean Cost Burden B", f"{gB.mean():.4f}")

        cs, cp = st.columns(2)
        cs.metric("U Statistic", f"{stat_ab:,.2f}")
        cp.metric("P-Value", f"{pval_ab:.6f}", delta="Tolak H₀ ✅" if pval_ab < 0.05 else "Gagal Tolak H₀ ❌")

        if pval_ab < 0.05:
            st.success(f"✅ **Tolak H₀** — p={pval_ab:.4f} < 0.05. Bank Sampah & TPS 3R menghasilkan cost_burden lebih rendah secara signifikan!")
            st.info("💡 Ini konsisten dengan misi Bank Sampah yang berfokus pada sampah bernilai ekonomi tinggi (kertas, botol, logam) yang memiliki cost_burden rendah.")
        else:
            st.warning(f"❌ **Gagal tolak H₀** — p={pval_ab:.4f} ≥ 0.05")

        col5, col6 = st.columns(2)

        with col5:
            hist_data = pd.DataFrame({
                'Cost Burden': list(gA) + list(gB),
                'Grup': ['Group A (Terstruktur)'] * len(gA) + ['Group B (Konvensional)'] * len(gB)
            })
            fig_ab_hist = px.histogram(
                hist_data, x='Cost Burden', color='Grup',
                barmode='overlay', opacity=0.65,
                color_discrete_map={'Group A (Terstruktur)': '#2ecc71', 'Group B (Konvensional)': '#e74c3c'},
                title='Distribusi Cost Burden: Group A vs Group B',
                nbins=10
            )
            fig_ab_hist.add_vline(x=gA.mean(), line_dash='dash', line_color='#27ae60',
                                   annotation_text=f'Mean A = {gA.mean():.2f}', annotation_position='top left')
            fig_ab_hist.add_vline(x=gB.mean(), line_dash='dash', line_color='#c0392b',
                                   annotation_text=f'Mean B = {gB.mean():.2f}', annotation_position='top right')
            fig_ab_hist.update_layout(height=420, title_font_size=14)
            st.plotly_chart(fig_ab_hist, use_container_width=True)

        with col6:
            fig_ab_box = go.Figure()
            fig_ab_box.add_trace(go.Box(y=gA.values, name='Group A<br>(Terstruktur)', marker_color='#2ecc71',
                                         boxmean=True, fillcolor='rgba(46,204,113,0.3)'))
            fig_ab_box.add_trace(go.Box(y=gB.values, name='Group B<br>(Konvensional)', marker_color='#e74c3c',
                                         boxmean=True, fillcolor='rgba(231,76,60,0.3)'))
            fig_ab_box.update_layout(
                title=f'Boxplot A/B Testing<br><sup>p-value = {pval_ab:.6f} | {"Tolak H₀" if pval_ab < 0.05 else "Gagal Tolak H₀"}</sup>',
                height=420, yaxis_title='Cost Burden', showlegend=True, title_font_size=14
            )
            st.plotly_chart(fig_ab_box, use_container_width=True)

        # Rata-rata per metode
        st.markdown('<div class="section-header">Rata-rata Cost Burden per Metode Pengumpulan</div>', unsafe_allow_html=True)
        metode_bb = df_f.groupby('Metode Pengumpulan')['cost_burden'].agg(['mean', 'std', 'count']).reset_index()
        metode_bb.columns = ['Metode', 'Mean', 'Std', 'Count']
        metode_bb['Grup'] = metode_bb['Metode'].apply(
            lambda x: 'Group A (Terstruktur)' if x in ['Bank Sampah', 'TPS 3R'] else 'Group B (Konvensional)'
        )
        metode_bb = metode_bb.sort_values('Mean')

        fig_metode = px.bar(
            metode_bb, y='Metode', x='Mean',
            color='Grup', color_discrete_map={'Group A (Terstruktur)': '#2ecc71', 'Group B (Konvensional)': '#e74c3c'},
            orientation='h', title='Rata-rata Cost Burden per Metode Pengumpulan',
            text='Mean', error_x='Std'
        )
        fig_metode.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_metode.add_vline(x=0, line_dash='dash', line_color='black', opacity=0.5)
        fig_metode.update_layout(height=380, title_font_size=14,
                                  xaxis_title='Mean Cost Burden', yaxis_title='')
        st.plotly_chart(fig_metode, use_container_width=True)

        result_color = "#f0fdf4" if pval_ab < 0.05 else "#fef2f2"
        border_color = "#86efac" if pval_ab < 0.05 else "#fca5a5"
        conclusion = "✅ <strong>TOLAK H₀</strong> — Terbukti secara statistik bahwa metode <strong>Bank Sampah / TPS 3R</strong> menghasilkan cost_burden <strong>lebih rendah</strong> secara signifikan." if pval_ab < 0.05 else "❌ <strong>GAGAL TOLAK H₀</strong> — Tidak cukup bukti statistik."

        st.markdown(f"""
        <div style="background:{result_color}; border:1.5px solid {border_color}; border-radius:10px; padding:1rem 1.4rem; margin-top:1rem;">
        <strong>Kesimpulan Statistik:</strong> {conclusion}<br><br>
        <strong>Implikasi Bisnis:</strong> Metode terstruktur mendorong pemilahan sampah sejak sumber (hulu),
        sehingga komoditas yang masuk ke industri lebih homogen dan bersih — mengurangi biaya sortir ulang di pabrik.
        <br><br>
        <strong>Rekomendasi:</strong> Perkuat investasi pada ekosistem <em>Bank Sampah & TPS 3R</em> dan kurangi ketergantungan pada metode Drop Box pasif.
        </div>""", unsafe_allow_html=True)

    else:
        st.warning("Data tidak cukup untuk A/B Testing. Silakan perluas filter metode pengumpulan.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#999; font-size:0.82rem; padding:1rem 0;">
    ♻️ <strong>Trashkara</strong> — Intelligent Waste Classifier & Generative Upcycling Assistant<br>
    CC26-PSU013 | Data Science 2026 | Dashboard dibuat dengan Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
