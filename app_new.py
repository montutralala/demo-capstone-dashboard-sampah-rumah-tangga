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
    try:
        df = pd.read_csv("data/dataset_trashkara_processed.csv")
        return df
    except FileNotFoundError:
        pass

    df_raw = pd.read_csv("data/dataset_trashkara_clean.csv")
    df = df_raw.copy()

    # Binary encoding
    for col in ['Dapat Terurai', 'Daur Ulang', 'Nilai Jual']:
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

    emisi_n = (df['Emisi CO2e (kg)'] - df['Emisi CO2e (kg)'].min()) / \
              (df['Emisi CO2e (kg)'].max() - df['Emisi CO2e (kg)'].min())
    urai_n  = np.log1p(df['waktu_urai_bulan'].fillna(0))
    urai_n  = (urai_n - urai_n.min()) / (urai_n.max() - urai_n.min())

    df['env_impact_score']     = emisi_n * 0.4 + urai_n * 0.4 + (1 - df['Dapat Terurai'].fillna(0)) * 0.2
    df['econ_burden_score']    = (df['biaya_proses_encoded'].fillna(2)/5)*0.6 + (1 - df['nilai_jual_encoded'].fillna(2)/5)*0.4
    df['recyclability_index']  = (df['Daur Ulang'].fillna(0)*0.4 +
                                   (df['nilai_jual_encoded'].fillna(1)/5)*0.35 +
                                   (1 - df['kesulitan_encoded'].fillna(2)/5)*0.25)
    return df

df = load_data()

CAT_COLOR = {'Anorganik': '#3498db', 'B3': '#e74c3c', 'Organik': '#2ecc71'}
KES_ORDER  = ['Mudah', 'Sedang', 'Sulit', 'Sangat Sulit']

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>♻️ Trashkara — Dashboard Analisis Data Sampah</h1>
    <p>Intelligent Waste Classifier & Generative Upcycling Assistant | CC26-PSU013 | Data Science 2026</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Filter Data")

    kategori_filter = st.multiselect(
        "Kategori Sampah",
        options=df['Kategori'].unique().tolist(),
        default=df['Kategori'].unique().tolist()
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

df_filtered = df[
    df['Kategori'].isin(kategori_filter) &
    df['Kesulitan Daur Ulang'].isin(kesulitan_filter) &
    df['Metode Pengumpulan'].isin(metode_filter) &
    df['Emisi CO2e (kg)'].between(emisi_range[0], emisi_range[1])
]

# ─────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
kpi_data = [
    (c1, "Total Data", f"{len(df_filtered):,}", "baris tersaring"),
    (c2, "Jenis Sampah", f"{df_filtered['Nama Sampah'].nunique()}", "dari 47 total"),
    (c3, "Rata-rata Emisi", f"{df_filtered['Emisi CO2e (kg)'].mean():.2f} kg", "CO₂e per sampel"),
    (c4, "Rata-rata Beban Biaya", f"{df_filtered['beban_biaya_score'].mean():.2f}", "skor (4–10)"),
    (c5, "High-Cost Burden", f"{(df_filtered['biaya_proses_encoded']==5).sum():,}", "sampel kritis"),
]
for col, label, val, sub in kpi_data:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{val}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview Dataset",
    "🌿 BQ1 — Emisi CO₂e",
    "💰 BQ2 — Struktur Biaya",
    "🔬 Analisis Lanjutan",
    "🧪 A/B Testing"
])

# ══════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Tentang Proyek Trashkara</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#f0fdf4; border:1.5px solid #86efac; border-radius:12px; padding:1.2rem 1.6rem; margin-bottom:1.2rem; font-size:0.92rem; color:#14532d; line-height:1.7;">
    <strong>🌱 Trashkara</strong> adalah platform <em>Intelligent Waste Classifier & Generative Upcycling Assistant</em>
    yang dikembangkan oleh tim <strong>CC26-PSU013</strong> dalam program Data Science 2026.<br><br>
    Dashboard ini menyajikan analisis komprehensif terhadap <strong>dataset sampah rumah tangga Indonesia</strong> —
    mencakup karakteristik material, jejak karbon (emisi CO₂e), potensi daur ulang, dan struktur beban biaya pengelolaan.
    Tujuan utama: mendukung pengambilan keputusan berbasis data untuk <strong>pengelolaan sampah yang lebih efisien dan berkelanjutan</strong>.
    </div>
    """, unsafe_allow_html=True)

    # Ringkasan 3 pertanyaan bisnis
    st.markdown('<div class="section-header">Pertanyaan Bisnis yang Dijawab</div>', unsafe_allow_html=True)
    bq_col1, bq_col2, bq_col3 = st.columns(3)
    with bq_col1:
        st.markdown("""
        <div style="background:white; border:1px solid #d1fae5; border-radius:10px; padding:1rem 1.2rem; height:140px;">
            <div style="font-size:1.4rem;">🌿</div>
            <div style="font-weight:600; color:#0f4c35; font-size:0.9rem; margin:0.3rem 0;">BQ1 — Emisi CO₂e</div>
            <div style="font-size:0.82rem; color:#555;">Kategori mana yang menyumbang emisi tertinggi & korelasi dengan kesulitan daur ulang?</div>
        </div>""", unsafe_allow_html=True)
    with bq_col2:
        st.markdown("""
        <div style="background:white; border:1px solid #d1fae5; border-radius:10px; padding:1rem 1.2rem; height:140px;">
            <div style="font-size:1.4rem;">💰</div>
            <div style="font-weight:600; color:#0f4c35; font-size:0.9rem; margin:0.3rem 0;">BQ2 — Struktur Biaya</div>
            <div style="font-size:0.82rem; color:#555;">Distribusi beban biaya lintas kategori & kuadran high-cost burden?</div>
        </div>""", unsafe_allow_html=True)
    with bq_col3:
        st.markdown("""
        <div style="background:white; border:1px solid #d1fae5; border-radius:10px; padding:1rem 1.2rem; height:140px;">
            <div style="font-size:1.4rem;">🧪</div>
            <div style="font-weight:600; color:#0f4c35; font-size:0.9rem; margin:0.3rem 0;">A/B Testing</div>
            <div style="font-size:0.82rem; color:#555;">Apakah metode pengumpulan terstruktur menghasilkan beban biaya lebih rendah?</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Gambaran Umum Dataset Trashkara 2026</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Komposisi Kategori
        cat_j = df_filtered.groupby('Kategori')['Nama Sampah'].nunique().reset_index()
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
        kes_cnt = df_filtered['Kesulitan Daur Ulang'].value_counts().reindex(KES_ORDER).reset_index()
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
            df_filtered, x='Emisi CO2e (kg)', nbins=30,
            color='Kategori', color_discrete_map=CAT_COLOR,
            title='Distribusi Emisi CO₂e per Kategori',
            barmode='overlay', opacity=0.7
        )
        mean_val = df_filtered['Emisi CO2e (kg)'].mean()
        fig_hist.add_vline(x=mean_val, line_dash='dash', line_color='#7f0000',
                           annotation_text=f'Mean = {mean_val:.2f} kg', annotation_position='top right')
        fig_hist.update_layout(height=360, title_font_size=14,
                                xaxis_title='Emisi CO₂e (kg)', yaxis_title='Frekuensi')
        st.plotly_chart(fig_hist, use_container_width=True)

    with col4:
        # Kategori Waktu Urai
        urai_cnt = df_filtered['kategori_urai'].value_counts().reset_index()
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

    # Insight ringkasan dataset
    st.markdown("""
    <div class="insight-box">
    <strong>📋 Ringkasan Dataset:</strong><br>
    • Dataset mencakup <strong>3 kategori utama</strong>: Organik, Anorganik, dan B3 (Bahan Berbahaya & Beracun).<br>
    • Material <strong>Organik</strong> mendominasi dari sisi emisi CO₂e namun paling mudah terurai secara alami.<br>
    • Sampah <strong>B3</strong> memiliki waktu urai terlama dan beban biaya pengelolaan tertinggi.<br>
    • Tingkat kesulitan daur ulang bervariasi dari <em>Mudah</em> hingga <em>Sangat Sulit</em> tergantung jenis material.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2: BQ1 — EMISI CO2E
# ══════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="bq-box">
    ❓ <strong>Pertanyaan Bisnis 1:</strong> Kategori sampah manakah yang menyumbang rata-rata emisi CO₂e tertinggi,
    dan apakah tingkat kesulitan daur ulang berkorelasi positif secara signifikan dengan besarnya emisi yang dihasilkan?
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        emisi_cat = df_filtered.groupby('Kategori')['Emisi CO2e (kg)'].mean().reset_index().sort_values('Emisi CO2e (kg)', ascending=False)
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

    with col2:
        emisi_kes = df_filtered.groupby('Kesulitan Daur Ulang')['Emisi CO2e (kg)'].mean().reindex(KES_ORDER).reset_index()
        emisi_kes.columns = ['Kesulitan', 'Mean Emisi']
        fig_kes_emisi = px.line(
            emisi_kes, x='Kesulitan', y='Mean Emisi',
            markers=True, title='Rata-rata Emisi CO₂e vs Tingkat Kesulitan Daur Ulang',
            line_shape='spline'
        )
        fig_kes_emisi.update_traces(line_color='#e74c3c', marker=dict(size=10, color='#c0392b'),
                                    text=[f'{v:.3f}' for v in emisi_kes['Mean Emisi']],
                                    textposition='top center', mode='lines+markers+text')
        fig_kes_emisi.update_layout(height=400, title_font_size=14,
                                     xaxis_title='Tingkat Kesulitan', yaxis_title='Rata-rata Emisi CO₂e (kg)')
        st.plotly_chart(fig_kes_emisi, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Box plot emisi per kategori
        fig_box = px.box(
            df_filtered, x='Kategori', y='Emisi CO2e (kg)',
            color='Kategori', color_discrete_map=CAT_COLOR,
            title='Distribusi Emisi CO₂e per Kategori (Box Plot)',
            points='outliers'
        )
        fig_box.update_layout(height=400, showlegend=False, title_font_size=14)
        st.plotly_chart(fig_box, use_container_width=True)

    with col4:
        # Violin plot
        fig_violin = px.violin(
            df_filtered, x='Kesulitan Daur Ulang', y='Emisi CO2e (kg)',
            color='Kesulitan Daur Ulang',
            category_orders={'Kesulitan Daur Ulang': KES_ORDER},
            title='Distribusi Emisi per Tingkat Kesulitan (Violin)',
            box=True, points=False
        )
        fig_violin.update_layout(height=400, showlegend=False, title_font_size=14)
        st.plotly_chart(fig_violin, use_container_width=True)

    # Spearman Correlation
    st.markdown('<div class="section-header">🔬 Uji Korelasi Spearman: Kesulitan Daur Ulang ↔ Emisi CO₂e</div>', unsafe_allow_html=True)

    df_corr = df_filtered[['kesulitan_encoded', 'Emisi CO2e (kg)']].dropna()
    rho, pval = spearmanr(df_corr['kesulitan_encoded'], df_corr['Emisi CO2e (kg)'])

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("Spearman ρ (rho)", f"{rho:.4f}", delta="Korelasi Positif" if rho > 0 else "Korelasi Negatif")
    with col_s2:
        st.metric("P-Value", f"{pval:.6f}", delta="Signifikan ✅" if pval < 0.05 else "Tidak Signifikan ❌")
    with col_s3:
        st.metric("Kesimpulan H₀", "TOLAK H₀" if pval < 0.05 else "GAGAL TOLAK H₀",
                  delta="p < 0.05" if pval < 0.05 else "p ≥ 0.05")

    # Top 10 jenis sampah emisi tertinggi
    st.markdown('<div class="section-header">Top 15 Jenis Sampah: Emisi CO₂e Tertinggi</div>', unsafe_allow_html=True)
    top_emisi = df_filtered.groupby(['Nama Sampah', 'Kategori'])['Emisi CO2e (kg)'].mean().reset_index()
    top_emisi = top_emisi.sort_values('Emisi CO2e (kg)', ascending=False).head(15)
    fig_top = px.bar(
        top_emisi, y='Nama Sampah', x='Emisi CO2e (kg)',
        color='Kategori', color_discrete_map=CAT_COLOR,
        orientation='h', title='Top 15 Jenis Sampah dengan Rata-rata Emisi CO₂e Tertinggi',
        text='Emisi CO2e (kg)'
    )
    fig_top.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_top.update_layout(height=500, title_font_size=14, yaxis={'categoryorder': 'total ascending'},
                           xaxis_title='Rata-rata Emisi CO₂e (kg)', yaxis_title='')
    st.plotly_chart(fig_top, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <strong>💡 Insight Pertanyaan Bisnis 1:</strong><br>
    • Kategori <strong>Organik</strong> menghasilkan emisi CO₂e tertinggi (~2.89 kg) akibat proses dekomposisi yang melepas gas metana (CH₄).<br>
    • Uji Spearman menunjukkan korelasi positif <strong>signifikan namun lemah</strong> (ρ ≈ 0.14, p &lt; 0.05).<br>
    • Artinya, kesulitan daur ulang bukan faktor utama emisi — <strong>karakteristik bahan dasar</strong> sampah jauh lebih menentukan.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3: BQ2 — STRUKTUR BIAYA
# ══════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="bq-box">
    ❓ <strong>Pertanyaan Bisnis 2:</strong> Bagaimana distribusi struktur beban biaya (gabungan nilai jual mentah dan biaya proses)
    lintas kategori utama, dan jenis sampah mana yang masuk kuadran biaya operasional tertinggi (high-cost burden)?
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Stacked bar distribusi struktur biaya
        komposisi = pd.crosstab(df_filtered['Kategori'], df_filtered['Struktur_Biaya'], normalize='index') * 100
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

    with col2:
        # Rata-rata komponen biaya
        ekon = df_filtered.groupby('Kategori')[['nilai_jual_encoded', 'biaya_proses_encoded']].mean().reset_index()

        fig_grouped = go.Figure()
        fig_grouped.add_trace(go.Bar(
            name='Nilai Jual Mentah', x=ekon['Kategori'], y=ekon['nilai_jual_encoded'],
            marker_color='#2c3e50', text=ekon['nilai_jual_encoded'].round(2),
            textposition='outside'
        ))
        fig_grouped.add_trace(go.Bar(
            name='Biaya Proses Industri', x=ekon['Kategori'], y=ekon['biaya_proses_encoded'],
            marker_color='#e67e22', text=ekon['biaya_proses_encoded'].round(2),
            textposition='outside'
        ))
        fig_grouped.update_layout(
            barmode='group', title='Rata-rata Nilai Jual vs Biaya Proses per Kategori (Skala 1–5)',
            height=420, title_font_size=14, yaxis_title='Encoded Value (1–5)',
            legend=dict(orientation='h', yanchor='bottom', y=-0.3)
        )
        st.plotly_chart(fig_grouped, use_container_width=True)

    # Total beban biaya per kategori
    col3, col4 = st.columns(2)

    with col3:
        bb_score = df_filtered['beban_biaya_score'].value_counts().sort_index().reset_index()
        bb_score.columns = ['Skor', 'Jumlah']
        fig_bb = px.bar(
            bb_score, x='Skor', y='Jumlah',
            title='Distribusi Total Skor Beban Biaya Operasional',
            color='Jumlah', color_continuous_scale='RdYlGn_r',
            text='Jumlah'
        )
        fig_bb.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_bb.update_layout(height=380, title_font_size=14,
                              xaxis_title='Beban Biaya Score (4–10)', yaxis_title='Frekuensi',
                              showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_bb, use_container_width=True)

    with col4:
        bb_cat = df_filtered.groupby('Kategori')['beban_biaya_score'].mean().reset_index().sort_values('beban_biaya_score')
        fig_hbar = px.bar(
            bb_cat, y='Kategori', x='beban_biaya_score',
            color='Kategori', color_discrete_map=CAT_COLOR,
            orientation='h', title='Rata-rata Total Beban Biaya per Kategori',
            text='beban_biaya_score'
        )
        fig_hbar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_hbar.update_layout(height=380, showlegend=False, title_font_size=14,
                                xaxis_title='Beban Biaya Score', yaxis_title='')
        st.plotly_chart(fig_hbar, use_container_width=True)

    # High-cost burden items
    st.markdown('<div class="section-header">⚠️ Jenis Sampah di Kuadran Biaya Operasional Tertinggi</div>', unsafe_allow_html=True)

    jenis_unik = df_filtered.drop_duplicates(subset=['Nama Sampah']).copy()
    high_cost = jenis_unik[jenis_unik['biaya_proses_encoded'] == 5].sort_values('beban_biaya_score', ascending=True)

    if len(high_cost) > 0:
        fig_hc = px.bar(
            high_cost, y='Nama Sampah', x='beban_biaya_score',
            color='Kategori', color_discrete_map=CAT_COLOR,
            orientation='h', title=f'Daftar {len(high_cost)} Jenis Sampah High-Cost Burden (Biaya Proses Level 5)',
            text='beban_biaya_score'
        )
        fig_hc.update_traces(texttemplate='Skor: %{text:.0f}', textposition='outside')
        fig_hc.update_layout(height=max(400, len(high_cost)*35+100), title_font_size=14,
                              xaxis_title='Total Skor Beban Biaya', yaxis_title='',
                              yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_hc, use_container_width=True)
    else:
        st.info("Tidak ada data high-cost burden pada filter yang dipilih.")

    st.markdown("""
    <div class="insight-box">
    <strong>💡 Insight Pertanyaan Bisnis 2:</strong><br>
    • Kategori <strong>B3 memiliki beban biaya tertinggi</strong> — 83.3% sampel berada pada kelompok Biaya Operasional Tinggi.<br>
    • <strong>15 jenis sampah</strong> masuk kuadran High-Cost Burden, didominasi sampah kaca dan elektronik.<br>
    • Puncak tertinggi: <strong>Kabel listrik bekas (skor 10)</strong> dan <strong>Aerosol (skor 8)</strong>.<br>
    • Rekomendasi: Terapkan kebijakan <em>Extended Producer Responsibility (EPR)</em> untuk mengurangi beban operasional industri.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4: ANALISIS LANJUTAN (FEATURE ENGINEERING)
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Feature Engineering: Skor Komposit Dampak Lingkungan & Ekonomi</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fe_stats = df_filtered.groupby('Kategori')[['env_impact_score', 'econ_burden_score', 'recyclability_index']].mean().round(3).reset_index()
        fe_stats.columns = ['Kategori', 'Env Impact Score', 'Econ Burden Score', 'Recyclability Index']

        fig_radar_data = []
        categories_radar = ['Env Impact Score', 'Econ Burden Score', 'Recyclability Index']
        fig_radar = go.Figure()
        for _, row in fe_stats.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Env Impact Score'], row['Econ Burden Score'], row['Recyclability Index']],
                theta=categories_radar,
                fill='toself',
                name=row['Kategori'],
                line_color=CAT_COLOR.get(row['Kategori'], '#888'),
                opacity=0.7
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title='Radar Chart: Profil Skor Komposit per Kategori',
            height=420, showlegend=True, title_font_size=14
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        st.markdown("**Rata-rata Skor Fitur Baru per Kategori**")
        st.dataframe(fe_stats, use_container_width=True, hide_index=True)

        st.markdown("""
        <div style="font-size:0.85rem; color:#555; margin-top:1rem; padding:0.8rem; background:#f9fafb; border-radius:8px;">
        <strong>Keterangan Skala:</strong><br>
        • <em>Env Impact Score</em>: 0 = Aman, 1 = Sangat Merusak Lingkungan<br>
        • <em>Econ Burden Score</em>: 0 = Menguntungkan, 1 = Beban Ekonomi Berat<br>
        • <em>Recyclability Index</em>: 0 = Sangat Sulit, 1 = Sangat Mudah Didaur Ulang
        </div>""", unsafe_allow_html=True)

    # Bubble Chart
    st.markdown('<div class="section-header">Bubble Chart: Dampak Lingkungan vs Beban Ekonomi</div>', unsafe_allow_html=True)

    fe_agg = df_filtered.groupby(['Nama Sampah', 'Kategori']).agg(
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

    mean_econ = fe_agg['econ'].mean()
    mean_env  = fe_agg['env'].mean()
    fig_bubble.add_hline(y=mean_env, line_dash='dash', line_color='gray', opacity=0.5)
    fig_bubble.add_vline(x=mean_econ, line_dash='dot', line_color='gray', opacity=0.5)

    fig_bubble.add_annotation(x=0.85, y=0.85, text="🔴 Kuadran Kritis", showarrow=False,
                               font=dict(color='#e74c3c', size=11))
    fig_bubble.add_annotation(x=0.2, y=0.2, text="🟢 Kuadran Emas", showarrow=False,
                               font=dict(color='#27ae60', size=11))

    fig_bubble.update_layout(height=520, title_font_size=14)
    st.plotly_chart(fig_bubble, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <strong>💡 Insight Feature Engineering:</strong><br>
    • <strong>B3 (Merah)</strong> — paling kritis: beban ekonomi tertinggi (~0.807) dan dampak lingkungan tertinggi (~0.597).<br>
    • <strong>Organik (Hijau)</strong> — paling aman: beban biaya termurah (~0.293) dan recyclability tertinggi (~0.817).<br>
    • <strong>Anorganik (Biru)</strong> — posisi menengah, sebaran luas karena variasi bahan yang beragam.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 5: A/B TESTING
# ══════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div class="bq-box">
    🧪 <strong>Hipotesis A/B Testing:</strong><br>
    H₁: Metode pengumpulan terstruktur (Bank Sampah / TPS 3R) menghasilkan beban biaya yang <em>lebih rendah secara signifikan</em>
    dibandingkan metode konvensional (Pengepul / Drop Box / Pickup Door-to-Door).<br>
    <em>Uji: Mann-Whitney U | One-Tailed | α = 0.05</em>
    </div>""", unsafe_allow_html=True)

    gA = df_filtered[df_filtered['Metode Pengumpulan'].isin(['Bank Sampah', 'TPS 3R'])]['beban_biaya_score'].dropna()
    gB = df_filtered[df_filtered['Metode Pengumpulan'].isin(['Pengepul', 'Drop Box', 'Pickup Door-to-Door'])]['beban_biaya_score'].dropna()

    if len(gA) > 0 and len(gB) > 0:
        stat_ab, pval_ab = mannwhitneyu(gA, gB, alternative='less')

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Group A (Terstruktur)", f"{gA.mean():.4f}", delta=f"n = {len(gA):,}")
        with col2:
            st.metric("Mean Group B (Konvensional)", f"{gB.mean():.4f}", delta=f"n = {len(gB):,}")
        with col3:
            st.metric("U-Statistic", f"{stat_ab:,.2f}")
        with col4:
            st.metric("P-Value", f"{pval_ab:.6f}", delta="Tolak H₀ ✅" if pval_ab < 0.05 else "Gagal Tolak H₀ ❌")

        col5, col6 = st.columns(2)

        with col5:
            hist_data = pd.DataFrame({
                'Beban Biaya Score': list(gA) + list(gB),
                'Grup': ['Group A (Terstruktur)'] * len(gA) + ['Group B (Konvensional)'] * len(gB)
            })
            fig_ab_hist = px.histogram(
                hist_data, x='Beban Biaya Score', color='Grup',
                barmode='overlay', opacity=0.65,
                color_discrete_map={'Group A (Terstruktur)': '#2ecc71', 'Group B (Konvensional)': '#e74c3c'},
                title='Distribusi Beban Biaya: Group A vs Group B',
                nbins=12
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
                height=420, yaxis_title='Beban Biaya Score', showlegend=True, title_font_size=14
            )
            st.plotly_chart(fig_ab_box, use_container_width=True)

        # Metode pengumpulan bar chart
        st.markdown('<div class="section-header">Rata-rata Beban Biaya per Metode Pengumpulan</div>', unsafe_allow_html=True)
        metode_bb = df_filtered.groupby('Metode Pengumpulan')['beban_biaya_score'].agg(['mean', 'std', 'count']).reset_index()
        metode_bb.columns = ['Metode', 'Mean', 'Std', 'Count']
        metode_bb['Grup'] = metode_bb['Metode'].apply(
            lambda x: 'Group A (Terstruktur)' if x in ['Bank Sampah', 'TPS 3R'] else 'Group B (Konvensional)'
        )
        metode_bb = metode_bb.sort_values('Mean')

        fig_metode = px.bar(
            metode_bb, y='Metode', x='Mean',
            color='Grup', color_discrete_map={'Group A (Terstruktur)': '#2ecc71', 'Group B (Konvensional)': '#e74c3c'},
            orientation='h', title='Rata-rata Beban Biaya per Metode Pengumpulan',
            text='Mean', error_x='Std'
        )
        fig_metode.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_metode.update_layout(height=380, title_font_size=14,
                                  xaxis_title='Mean Beban Biaya Score', yaxis_title='')
        st.plotly_chart(fig_metode, use_container_width=True)

        result_color = "#f0fdf4" if pval_ab < 0.05 else "#fef2f2"
        border_color = "#86efac" if pval_ab < 0.05 else "#fca5a5"
        conclusion = "✅ <strong>TOLAK H₀</strong> — Terbukti secara statistik bahwa metode <strong>Bank Sampah / TPS 3R</strong> menghasilkan beban biaya <strong>lebih rendah</strong> secara signifikan dibandingkan metode konvensional." if pval_ab < 0.05 else "❌ <strong>GAGAL TOLAK H₀</strong> — Tidak cukup bukti statistik untuk menyatakan metode terstruktur lebih hemat."

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
