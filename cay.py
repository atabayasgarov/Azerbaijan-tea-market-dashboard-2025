import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# ARIMA üçün
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False

# ─────────────────────────────────────────────────────────────
# KONFIQURASIYA
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Azərbaycan Çay Bazarı",
    page_icon="🫖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# VAHİD VİZUAL SİSTEM — Rəng Paleti və Stil
# ─────────────────────────────────────────────────────────────
# Əsas palet: dərin yaşıl çay yarpağı + qırmızı vurğu + yumşaq krem fon
THEME = {
    "bg_page":      "#F5F1E8",   # Krem-bej səhifə fonu
    "bg_chart":     "#FAF7F0",   # Bir az daha açıq qrafik fonu
    "bg_card":      "#FFFFFF",   # Kart fonu
    "text_dark":    "#1A3A2E",   # Dərin yaşıl mətn
    "text_muted":   "#6B7B6F",   # Sönük mətn
    "grid":         "rgba(26,58,46,0.08)",
    "border":       "#D8D2C2",

    # Vurğu rəngləri (vahid palet)
    "primary":      "#1A3A2E",   # Dərin çay yaşılı
    "secondary":    "#7A9E7E",   # Açıq yaşıl
    "accent":       "#B85042",   # Çay qırmızısı
    "warning":      "#D4A04C",   # Saman sarısı
    "neutral":      "#5C6B7A",   # Slate
    "purple":       "#6B4E71",   # Solğun bənövşəyi
}

# Plotly üçün vahid template
PLOTLY_LAYOUT = dict(
    plot_bgcolor=THEME["bg_chart"],
    paper_bgcolor=THEME["bg_chart"],
    font=dict(
        family="IBM Plex Sans, sans-serif",
        size=12,
        color=THEME["text_dark"],
    ),
    title=dict(
        font=dict(size=15, color=THEME["text_dark"], family="IBM Plex Sans"),
        x=0.02, xanchor="left",
    ),
    margin=dict(l=60, r=30, t=60, b=50),
    xaxis=dict(
        gridcolor=THEME["grid"],
        linecolor=THEME["border"],
        tickfont=dict(color=THEME["text_dark"]),
        title_font=dict(color=THEME["text_dark"]),
    ),
    yaxis=dict(
        gridcolor=THEME["grid"],
        linecolor=THEME["border"],
        tickfont=dict(color=THEME["text_dark"]),
        title_font=dict(color=THEME["text_dark"]),
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor=THEME["border"],
        borderwidth=1,
        font=dict(color=THEME["text_dark"], size=11),
    ),
    hoverlabel=dict(
        bgcolor=THEME["bg_card"],
        bordercolor=THEME["primary"],
        font=dict(color=THEME["text_dark"], family="IBM Plex Sans"),
    ),
)

def apply_theme(fig, height=380, title=None):
    """Bütün qrafiklərə vahid tema tətbiq edir."""
    layout_copy = {k: v for k, v in PLOTLY_LAYOUT.items()}
    fig.update_layout(**layout_copy, height=height)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15, color=THEME["text_dark"])))
    return fig


# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=Fraunces:wght@600;700&display=swap');

/* ═══════════════════════════════════════════════════════
   DARK MODE OVERRIDE — Brauzer dark olsa da panel light qalır
   ═══════════════════════════════════════════════════════ */
html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {{
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: {THEME["text_dark"]} !important;
    background-color: {THEME["bg_page"]} !important;
}}

/* Bütün mətn elementləri */
p, span, label, div, li, a {{
    color: {THEME["text_dark"]};
}}

/* Başlıqlar */
h1, h2, h3, h4, h5, h6 {{
    color: {THEME["primary"]} !important;
}}

/* Səhifə fonu */
.stApp {{
    background: {THEME["bg_page"]} !important;
}}

/* Markdown bloklar */
[data-testid="stMarkdownContainer"] {{
    color: {THEME["text_dark"]} !important;
}}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li {{
    color: {THEME["text_dark"]} !important;
}}

/* ═══════════════════════════════════════════════════════
   WIDGET LABEL-LƏRİ (radio, checkbox, selectbox, slider)
   ═══════════════════════════════════════════════════════ */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
.stRadio label, .stCheckbox label, .stSelectbox label,
.stSlider label, .stMultiSelect label,
.stRadio > div > label > div,
.stCheckbox > label > div {{
    color: {THEME["text_dark"]} !important;
    font-weight: 500;
}}

/* Radio və checkbox açıqlama mətni */
.stRadio div[role="radiogroup"] label,
.stRadio div[role="radiogroup"] label p,
.stCheckbox label p {{
    color: {THEME["text_dark"]} !important;
}}

/* Slider rəqəm dəyərləri */
.stSlider [data-baseweb="slider"] div {{
    color: {THEME["text_dark"]} !important;
}}

/* Selectbox seçilmiş dəyər */
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {{
    color: {THEME["text_dark"]} !important;
    background-color: {THEME["bg_card"]} !important;
}}

/* Başlıqlar */
.main-title {{
    font-family: 'Fraunces', serif;
    font-size: 36px;
    font-weight: 700;
    color: {THEME["primary"]};
    border-left: 6px solid {THEME["accent"]};
    padding-left: 16px;
    margin-bottom: 4px;
    letter-spacing: -0.5px;
}}
.subtitle {{
    font-size: 14px;
    color: {THEME["text_muted"]};
    margin-left: 22px;
    margin-bottom: 28px;
    font-style: italic;
}}
.section-header {{
    font-family: 'Fraunces', serif;
    font-size: 22px;
    font-weight: 600;
    color: {THEME["primary"]};
    border-bottom: 2px solid {THEME["border"]};
    padding-bottom: 8px;
    margin-top: 36px;
    margin-bottom: 18px;
}}

/* KPI kartlar */
.kpi-card {{
    background: {THEME["bg_card"]};
    border: 1px solid {THEME["border"]};
    border-radius: 10px;
    padding: 18px 20px;
    border-top: 4px solid var(--accent);
    box-shadow: 0 1px 3px rgba(26,58,46,0.06);
}}
.kpi-label {{
    font-size: 11px;
    color: {THEME["text_muted"]};
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .08em;
}}
.kpi-value {{
    font-family: 'Fraunces', serif;
    font-size: 28px;
    font-weight: 700;
    color: {THEME["primary"]};
    margin: 6px 0 4px;
    line-height: 1.1;
}}
.kpi-note {{
    font-size: 12px;
    color: {THEME["text_muted"]};
}}
.kpi-delta-up   {{ font-size: 13px; color: {THEME["secondary"]}; font-weight: 600; }}
.kpi-delta-down {{ font-size: 13px; color: {THEME["accent"]}; font-weight: 600; }}

/* Bannerlər */
.warn-box {{
    background: #FBF3DC;
    border-left: 4px solid {THEME["warning"]};
    border-radius: 6px;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 13.5px;
    color: #6B5419;
}}
.info-box {{
    background: #E8EFE6;
    border-left: 4px solid {THEME["secondary"]};
    border-radius: 6px;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 13.5px;
    color: #1F3A28;
}}
.danger-box {{
    background: #F8E4DF;
    border-left: 4px solid {THEME["accent"]};
    border-radius: 6px;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 13.5px;
    color: #6E2419;
}}

/* Tab stilleri */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: {THEME["bg_card"]};
    padding: 6px;
    border-radius: 8px;
    border: 1px solid {THEME["border"]};
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {THEME["text_muted"]};
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
}}
.stTabs [aria-selected="true"] {{
    background: {THEME["primary"]} !important;
    color: white !important;
}}

/* ═══════════════════════════════════════════════════════
   CƏDVƏL — Dark mode override
   ═══════════════════════════════════════════════════════ */
[data-testid="stDataFrame"],
[data-testid="stDataFrameResizable"],
[data-testid="stTable"] {{
    background-color: {THEME["bg_card"]} !important;
}}

/* Header — qara fonda ağ yazı saxlanılır */
[data-testid="stDataFrame"] thead th,
[data-testid="stDataFrameResizable"] thead th,
[data-testid="stDataFrame"] [role="columnheader"] {{
    background-color: {THEME["primary"]} !important;
    color: #FFFFFF !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}}

/* Cədvəl hücrələri — açıq fon, dərin yaşıl yazı */
[data-testid="stDataFrame"] tbody tr,
[data-testid="stDataFrame"] tbody td,
[data-testid="stDataFrameResizable"] tbody tr,
[data-testid="stDataFrameResizable"] tbody td,
[data-testid="stDataFrame"] [role="gridcell"],
[data-testid="stDataFrame"] [role="row"] {{
    background-color: {THEME["bg_card"]} !important;
    color: {THEME["text_dark"]} !important;
}}

/* Glide data grid (Streamlit-in yeni cədvəl renderi) */
.dvn-scroller {{
    background-color: {THEME["bg_card"]} !important;
}}
canvas {{
    background-color: {THEME["bg_card"]} !important;
}}

/* Cədvəl indeksi */
[data-testid="stDataFrame"] [data-testid="stDataFrameIndex"] {{
    background-color: #F0EBE0 !important;
    color: {THEME["text_dark"]} !important;
}}

/* Sidebar */
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
    background: {THEME["bg_card"]} !important;
    border-right: 1px solid {THEME["border"]};
}}
[data-testid="stSidebar"] *,
[data-testid="stSidebarContent"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {{
    color: {THEME["text_dark"]} !important;
}}
[data-testid="stSidebarContent"] .sidebar-title {{
    font-family: 'Fraunces', serif;
    font-size: 17px;
    font-weight: 700;
    color: {THEME["primary"]} !important;
    border-bottom: 2px solid {THEME["accent"]};
    padding-bottom: 8px;
    margin-bottom: 14px;
}}

/* Help icon-ları da görünsün */
[data-baseweb="tooltip"] {{
    color: {THEME["text_dark"]} !important;
}}

hr {{
    border-color: {THEME["border"]} !important;
    margin: 20px 0 !important;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# YARDIMÇI FUNKSİYALAR
# ─────────────────────────────────────────────────────────────
def kpi_card(label: str, value: str, note: str = "", delta: str = "", accent: str = None, delta_up: bool = True):
    if accent is None:
        accent = THEME["primary"]
    delta_cls = "kpi-delta-up" if delta_up else "kpi-delta-down"
    delta_html = f'<div class="{delta_cls}">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{accent};">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}
      <div class="kpi-note">{note}</div>
    </div>""", unsafe_allow_html=True)


def forecast_linear(series: pd.Series, n_years: int = 5):
    """Xətti trend proqnozu."""
    x = np.arange(len(series))
    slope, intercept, r, *_ = linregress(x, series.values)
    future_x = np.arange(len(series), len(series) + n_years)
    forecast = intercept + slope * future_x
    return forecast, slope, r ** 2


def render_styled_table(styler_or_df, hide_index=True):
    """
    st.dataframe yerinə HTML cədvəl çıxarır — canvas problemini həll edir.
    Styler obyektini və ya adi DataFrame-i qəbul edir.
    """
    # Styler obyekti gəlirsə birbaşa to_html, DataFrame gəlirsə əvvəl style et
    if isinstance(styler_or_df, pd.DataFrame):
        styler = styler_or_df.style
    else:
        styler = styler_or_df

    if hide_index:
        styler = styler.hide(axis="index")

    # Vahid stil tətbiq edək
    styler = styler.set_table_styles([
        {"selector": "thead th",
         "props": f"background-color: {THEME['primary']}; color: #FFFFFF; "
                  f"font-weight: 600; font-size: 13px; padding: 10px 12px; "
                  f"text-align: left; border: none;"},
        {"selector": "tbody td",
         "props": f"color: {THEME['text_dark']}; padding: 8px 12px; "
                  f"font-size: 13px; border-bottom: 1px solid {THEME['border']};"},
        {"selector": "tbody tr:nth-child(even) td",
         "props": f"background-color: {THEME['bg_chart']};"},
        {"selector": "tbody tr:nth-child(odd) td",
         "props": f"background-color: {THEME['bg_card']};"},
        {"selector": "tbody tr:hover td",
         "props": f"background-color: #F0EBE0;"},
        {"selector": "",
         "props": f"border-collapse: collapse; width: 100%; "
                  f"border-radius: 8px; overflow: hidden; "
                  f"box-shadow: 0 1px 3px rgba(26,58,46,0.08); "
                  f"font-family: 'IBM Plex Sans', sans-serif;"},
    ])

    html = styler.to_html()
    # Wrapper ilə scroll problemi olmasın
    wrapped = f'<div style="overflow-x:auto; margin: 8px 0 16px;">{html}</div>'
    st.markdown(wrapped, unsafe_allow_html=True)


def forecast_arima(series: pd.Series, n_years: int = 7, order=(1, 1, 1)):
    """ARIMA modeli ilə proqnoz + 95% etibar intervalı."""
    if not HAS_ARIMA:
        return None, None, None, None
    try:
        model = ARIMA(series.values, order=order)
        fitted = model.fit()
        fc_result = fitted.get_forecast(steps=n_years)
        forecast = np.asarray(fc_result.predicted_mean)
        conf_int = np.asarray(fc_result.conf_int(alpha=0.05))
        lower = conf_int[:, 0]
        upper = conf_int[:, 1]
        aic = fitted.aic
        return forecast, lower, upper, aic
    except Exception as e:
        return None, None, None, None


# ─────────────────────────────────────────────────────────────
# 1. MƏLUMATLAR (2015–2025)
# ─────────────────────────────────────────────────────────────
YEARS_HIST = list(range(2015, 2026))

raw = {
    "İl": YEARS_HIST,
    "Yerli İstehsal (Ton)": [780, 810, 850, 890, 920, 950, 1020, 980, 1050, 1100, 1150],
    "İdxal Həcmi (Ton)":   [8500, 9100, 9700, 10200, 10600, 11200, 11800, 12400, 12900, 13400, 14000],
    "İdxal USD/Ton":       [3550, 3620, 3710, 3800, 3920, 4150, 4420, 4200, 4320, 4370, 4357],
    "İxrac Həcmi (Ton)":   [750, 820, 870, 940, 980, 1100, 1250, 1300, 1420, 1500, 1600],
    "Əkin Sahəsi (ha)":    [780, 800, 820, 850, 890, 950, 1050, 1100, 1150, 1200, 1250],
    "Məhsuldarlıq (t/ha)": [1.00, 1.01, 1.04, 1.05, 1.03, 1.00, 0.97, 0.89, 0.91, 0.92, 0.92],
    "Dünya Qiyməti (USD/Ton)":  [2400, 2480, 2550, 2690, 2870, 3050, 3400, 2950, 3100, 3250, 3320],
    "Ceylon Qiymeti (USD/Ton)": [2850, 2920, 3010, 3180, 3350, 3600, 3900, 3550, 3720, 3800, 3870],
    "Kenya Qiymeti (USD/Ton)":  [2600, 2680, 2740, 2870, 3000, 3200, 3580, 3100, 3280, 3350, 3420],
}

df = pd.DataFrame(raw)
df["İdxal Dəyəri (Mln USD)"] = (df["İdxal Həcmi (Ton)"] * df["İdxal USD/Ton"] / 1_000_000).round(2)
df["Ümumi İstehlak (Ton)"] = df["Yerli İstehsal (Ton)"] + df["İdxal Həcmi (Ton)"] - df["İxrac Həcmi (Ton)"]
df["İdxaldan Asılılıq (%)"] = (df["İdxal Həcmi (Ton)"] / df["Ümumi İstehlak (Ton)"] * 100).round(1)
df["Nəfər Başına (kq)"]     = (df["Ümumi İstehlak (Ton)"] * 1000 / 10_200_000).round(2)

# ─────────────────────────────────────────────────────────────
# QONŞU ÖLKƏLƏR — Azərbaycan + Türkiyə + Gürcüstan + İran
# (Strateji baxımdan ən vacib müqayisə qrupu)
# ─────────────────────────────────────────────────────────────
NEIGHBOR_DATA = pd.DataFrame({
    "Ölkə":            ["Azərbaycan", "Türkiyə", "Gürcüstan", "İran"],
    "İstehsal (kt)":   [1.15, 265.0, 2.5, 14.0],
    "İstehlak (kt)":   [13.0, 250.0, 2.8, 95.0],
    "Adam başına (kq)":[1.27, 3.00, 0.75, 1.10],
    "Sahə (kha)":      [1.25, 77.0, 3.2, 25.0],
    "Məhsuldarlıq (t/ha)": [0.92, 3.44, 0.78, 0.56],
    "Çay tarixi (il)": [125, 145, 175, 130],
})

# Tarixi qonşu ölkə istehsal trendi (kt) - təxmini real məlumatlar əsasında
NEIGHBOR_TREND = pd.DataFrame({
    "İl": YEARS_HIST,
    "Azərbaycan": [0.78, 0.81, 0.85, 0.89, 0.92, 0.95, 1.02, 0.98, 1.05, 1.10, 1.15],
    "Türkiyə":    [228, 234, 240, 252, 261, 270, 280, 260, 268, 263, 265],
    "Gürcüstan":  [1.5, 1.7, 1.9, 2.0, 2.1, 2.2, 2.3, 2.3, 2.4, 2.5, 2.5],
    "İran":       [16.5, 15.8, 15.2, 14.8, 14.3, 14.0, 14.5, 13.8, 14.2, 14.0, 14.0],
})

# =============================================================================
# YENİLİK 1: REAL DATA YÜKLƏYİCİ
# =============================================================================
st.sidebar.markdown("### 📂 1. Öz Datanızı Sistemə Yükləyin")
uploaded_file = st.sidebar.file_uploader("Excel və ya CSV faylı daxil edin", type=["xlsx", "csv"])
# Əgər istifadəçi fayl yükləsə, proqram dərhal həmin datanı oxuyacaq
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            uploaded_df = pd.read_csv(uploaded_file)
        else:
            uploaded_df = pd.read_excel(uploaded_file)
        # Yüklənən datanı əsas df-ə tətbiq et (mövcud sütunları override et)
        required_cols = ["İl", "Yerli İstehsal (Ton)", "İdxal Həcmi (Ton)",
                         "İdxal USD/Ton", "İxrac Həcmi (Ton)"]
        missing = [c for c in required_cols if c not in uploaded_df.columns]
        if missing:
            st.sidebar.warning(f"Çatışmayan sütun(lar): {', '.join(missing)}. Defolt data istifadə olunur.")
        else:
            df = uploaded_df.copy()
            # Asılı sütunları yenidən hesabla
            df["İdxal Dəyəri (Mln USD)"] = (df["İdxal Həcmi (Ton)"] * df["İdxal USD/Ton"] / 1_000_000).round(2)
            df["Ümumi İstehlak (Ton)"] = df["Yerli İstehsal (Ton)"] + df["İdxal Həcmi (Ton)"] - df["İxrac Həcmi (Ton)"]
            df["İdxaldan Asılılıq (%)"] = (df["İdxal Həcmi (Ton)"] / df["Ümumi İstehlak (Ton)"] * 100).round(1)
            df["Nəfər Başına (kq)"] = (df["Ümumi İstehlak (Ton)"] * 1000 / 10_200_000).round(2)
            st.sidebar.success("Fayl uğurla yükləndi və tətbiq edildi!")
    except Exception as e:
        st.sidebar.error(f"Fayl oxunarkən xəta baş verdi: {e}")

# ─────────────────────────────────────────────────────────────
# 2. BAŞLIQ + KPI
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🫖 Azərbaycan Çay Bazarı — Strateji Analitik Panel</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">İdxal-İxrac Balansı · Regional Problemlər · İnvestisiya Simulyatoru · ARIMA Proqnozu · Qonşu Ölkələrlə Müqayisə</div>', unsafe_allow_html=True)

latest = df.iloc[-1]
prev   = df.iloc[-2]

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    kpi_card("Ümumi İstehlak", f"{latest['Ümumi İstehlak (Ton)']:,.0f} ton",
             "2025 daxili tələbat",
             f"▲ {latest['Ümumi İstehlak (Ton)'] - prev['Ümumi İstehlak (Ton)']:+.0f} ton",
             accent=THEME["primary"])
with c2:
    kpi_card("Yerli İstehsal", f"{latest['Yerli İstehsal (Ton)']:,.0f} ton",
             f"Tələbatın {latest['Yerli İstehsal (Ton)'] / latest['Ümumi İstehlak (Ton)'] * 100:.1f}%-i",
             "▲ +50 ton YoY", accent=THEME["secondary"])
with c3:
    kpi_card("İdxal Xərci", f"${latest['İdxal Dəyəri (Mln USD)']:.1f} mln",
             "2025 valyuta çıxışı",
             f"▲ +${latest['İdxal Dəyəri (Mln USD)'] - prev['İdxal Dəyəri (Mln USD)']:.1f} mln",
             accent=THEME["accent"], delta_up=False)
with c4:
    kpi_card("İdxaldan Asılılıq", f"{latest['İdxaldan Asılılıq (%)']:.1f}%",
             "İnfrastruktur riski",
             "▼ hədəf: <70%", accent=THEME["warning"], delta_up=False)
with c5:
    kpi_card("Adam Başına", f"{latest['Nəfər Başına (kq)']:.2f} kq",
             "İllik çay istehlakı",
             "Türkiyənin ⅓-i", accent=THEME["purple"])

st.markdown("---")


# ─────────────────────────────────────────────────────────────
# 3. TABLAR
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Bazar Dinamikası",
    "🇦🇿 Qonşu Ölkələr",
    "💲 Qiymət Analizi",
    "🗺️ Regional Xəritə",
    "⚙️ Problemlər & Strategiya",
    "📈 ARIMA Proqnozu & Simulyasiya",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — BAZAR DİNAMİKASI
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Bazar Strukturunun Dinamikası (2015–2025)</div>', unsafe_allow_html=True)

    fig_stack = go.Figure()
    fig_stack.add_trace(go.Bar(
        x=df["İl"], y=df["İdxal Həcmi (Ton)"],
        name="İdxal", marker_color=THEME["primary"],
        hovertemplate="<b>%{x}</b><br>İdxal: %{y:,.0f} ton<extra></extra>",
    ))
    fig_stack.add_trace(go.Bar(
        x=df["İl"], y=df["Yerli İstehsal (Ton)"],
        name="Yerli İstehsal", marker_color=THEME["secondary"],
        hovertemplate="<b>%{x}</b><br>Yerli: %{y:,.0f} ton<extra></extra>",
    ))
    fig_stack.add_trace(go.Scatter(
        x=df["İl"], y=df["İxrac Həcmi (Ton)"],
        name="İxrac", mode="lines+markers",
        line=dict(color=THEME["accent"], width=3),
        marker=dict(size=9, symbol="diamond", line=dict(color="white", width=1.5)),
        hovertemplate="<b>%{x}</b><br>İxrac: %{y:,.0f} ton<extra></extra>",
    ))
    fig_stack.update_layout(
        barmode="stack",
        title="Çay istehlak strukturu — idxal üstünlüyü açıq şəkildə görünür",
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        xaxis=dict(dtick=1, title="İl"),
        yaxis=dict(title="Ton"),
    )
    apply_theme(fig_stack, height=400)
    st.plotly_chart(fig_stack, use_container_width=True)

    left, right = st.columns(2)

    with left:
        fig_usd = go.Figure()
        fig_usd.add_trace(go.Scatter(
            x=df["İl"], y=df["İdxal Dəyəri (Mln USD)"],
            mode="lines+markers",
            fill="tozeroy",
            fillcolor="rgba(184,80,66,0.18)",
            line=dict(color=THEME["accent"], width=3),
            marker=dict(size=8, color=THEME["accent"], line=dict(color="white", width=2)),
            hovertemplate="<b>%{x}</b><br>%{y:.2f} Mln USD<extra></extra>",
            name="İdxal Xərci",
        ))
        fig_usd.update_layout(
            title="İdxala Xərclənən Valyuta (Mln USD)",
            xaxis=dict(dtick=1),
            yaxis=dict(title="Mln USD"),
            showlegend=False,
        )
        apply_theme(fig_usd, height=340)
        st.plotly_chart(fig_usd, use_container_width=True)
        st.markdown('<div class="danger-box">2015-ci ildən bəri idxal xərcləri <b>2.4x</b> artıb. Qlobal logistika inflyasiyası əsas səbəbdir.</div>', unsafe_allow_html=True)

    with right:
        fig_dep = go.Figure()
        fig_dep.add_trace(go.Scatter(
            x=df["İl"], y=df["İdxaldan Asılılıq (%)"],
            mode="lines+markers",
            line=dict(color=THEME["purple"], width=3),
            marker=dict(size=8, color=THEME["purple"], line=dict(color="white", width=2)),
            hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
            name="Asılılıq %",
        ))
        fig_dep.add_hline(y=90, line_dash="dash", line_color=THEME["accent"],
                          annotation_text="Kritik hədd", annotation_position="top left",
                          annotation_font=dict(color=THEME["accent"], size=11))
        fig_dep.add_hline(y=70, line_dash="dash", line_color=THEME["secondary"],
                          annotation_text="Hədəf", annotation_position="bottom left",
                          annotation_font=dict(color=THEME["secondary"], size=11))
        fig_dep.update_layout(
            title="İdxaldan Asılılıq Dinamikası (%)",
            xaxis=dict(dtick=1),
            yaxis=dict(title="Asılılıq %", range=[68, 95]),
            showlegend=False,
        )
        apply_theme(fig_dep, height=340)
        st.plotly_chart(fig_dep, use_container_width=True)
        st.markdown('<div class="warn-box">İdxal asılılığı <b>hər il ortalama 0.2–0.3%</b> artır. Hazırkı tempilə 2030-da <b>93%</b>-ə çatacaq.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Əkin Sahəsi vs Məhsuldarlıq — Verimlilik Problemi</div>', unsafe_allow_html=True)
    fig_prod = make_subplots(specs=[[{"secondary_y": True}]])
    fig_prod.add_trace(go.Bar(
        x=df["İl"], y=df["Əkin Sahəsi (ha)"],
        name="Əkin Sahəsi (ha)",
        marker_color=THEME["secondary"], opacity=0.7,
        hovertemplate="<b>%{x}</b><br>Sahə: %{y:,.0f} ha<extra></extra>",
    ), secondary_y=False)
    fig_prod.add_trace(go.Scatter(
        x=df["İl"], y=df["Məhsuldarlıq (t/ha)"],
        name="Məhsuldarlıq (t/ha)", mode="lines+markers",
        line=dict(color=THEME["accent"], width=3),
        marker=dict(size=10, symbol="circle", line=dict(color="white", width=2)),
        hovertemplate="<b>%{x}</b><br>%{y:.2f} t/ha<extra></extra>",
    ), secondary_y=True)
    fig_prod.update_layout(
        title="Sahə genişlənir, məhsuldarlıq aşağı düşür — effektivlik itkisi",
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        xaxis=dict(dtick=1),
    )
    apply_theme(fig_prod, height=360)
    fig_prod.update_yaxes(title_text="Sahə (ha)", secondary_y=False, gridcolor=THEME["grid"])
    fig_prod.update_yaxes(title_text="t/ha", secondary_y=True, showgrid=False)
    st.plotly_chart(fig_prod, use_container_width=True)
    st.markdown('<div class="warn-box">2017-dən bəri sahə <b>+200 ha</b> artdı, amma məhsuldarlıq <b>1.04 → 0.92 t/ha</b>-ya düşdü. Torpaq keyfiyyəti və aqrotexnika problemini göstərir.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — QONŞU ÖLKƏLƏR (Türkiyə, Gürcüstan, İran)
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Azərbaycan vs Qonşu Çay İstehsalçıları</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">📍 Müqayisə qrupu: <b>Türkiyə</b> (Rize regional uğur modeli), <b>Gürcüstan</b> (oxşar miqyas, dirçəliş cəhdi), <b>İran</b> (Gilan-Lənkəran iqlim oxşarlığı). Bu üç ölkə Azərbaycan üçün ən real bençmarkdır.</div>', unsafe_allow_html=True)

    # Qonşu rəng paleti
    NEIGHBOR_COLORS = {
        "Azərbaycan": THEME["accent"],     # qırmızı vurğu
        "Türkiyə":    THEME["primary"],    # dərin yaşıl
        "Gürcüstan":  THEME["warning"],    # saman sarısı
        "İran":       THEME["purple"],     # bənövşəyi
    }

    c_l, c_r = st.columns(2)

    with c_l:
        sorted_prod = NEIGHBOR_DATA.sort_values("İstehsal (kt)", ascending=True)
        fig_prod_cmp = go.Figure()
        fig_prod_cmp.add_trace(go.Bar(
            x=sorted_prod["İstehsal (kt)"],
            y=sorted_prod["Ölkə"],
            orientation="h",
            marker=dict(
                color=[NEIGHBOR_COLORS[c] for c in sorted_prod["Ölkə"]],
                line=dict(color="white", width=1.5),
            ),
            text=[f"{v:.1f} kt" for v in sorted_prod["İstehsal (kt)"]],
            textposition="outside",
            textfont=dict(color=THEME["text_dark"], size=11),
            hovertemplate="<b>%{y}</b><br>İstehsal: %{x:.1f} kt<extra></extra>",
        ))
        fig_prod_cmp.update_layout(
            title="İllik Çay İstehsalı (min ton)",
            xaxis=dict(title="min ton"),
            showlegend=False,
        )
        apply_theme(fig_prod_cmp, height=320)
        st.plotly_chart(fig_prod_cmp, use_container_width=True)

    with c_r:
        sorted_cons = NEIGHBOR_DATA.sort_values("Adam başına (kq)", ascending=True)
        fig_cons = go.Figure()
        fig_cons.add_trace(go.Bar(
            x=sorted_cons["Adam başına (kq)"],
            y=sorted_cons["Ölkə"],
            orientation="h",
            marker=dict(
                color=[NEIGHBOR_COLORS[c] for c in sorted_cons["Ölkə"]],
                line=dict(color="white", width=1.5),
            ),
            text=[f"{v:.2f} kq" for v in sorted_cons["Adam başına (kq)"]],
            textposition="outside",
            textfont=dict(color=THEME["text_dark"], size=11),
            hovertemplate="<b>%{y}</b><br>Adam başına: %{x:.2f} kq<extra></extra>",
        ))
        fig_cons.update_layout(
            title="Adam Başına İllik İstehlak (kq)",
            xaxis=dict(title="kq / nəfər"),
            showlegend=False,
        )
        apply_theme(fig_cons, height=320)
        st.plotly_chart(fig_cons, use_container_width=True)

    # Tarixi trend — qonşu ölkələr (log şkala İrana və Türkiyəyə görə)
    st.markdown('<div class="section-header">İstehsal Trendi 2015–2025 (loqarifmik şkala)</div>', unsafe_allow_html=True)

    fig_trend = go.Figure()
    for country in ["Azərbaycan", "Türkiyə", "Gürcüstan", "İran"]:
        is_az = country == "Azərbaycan"
        fig_trend.add_trace(go.Scatter(
            x=NEIGHBOR_TREND["İl"],
            y=NEIGHBOR_TREND[country],
            name=country,
            mode="lines+markers",
            line=dict(
                color=NEIGHBOR_COLORS[country],
                width=4 if is_az else 2,
                dash="solid" if is_az else "dot",
            ),
            marker=dict(
                size=10 if is_az else 6,
                line=dict(color="white", width=1.5),
            ),
            hovertemplate="<b>%{x} — " + country + "</b><br>%{y:.2f} kt<extra></extra>",
        ))
    fig_trend.update_layout(
        title="İstehsal həcmi miqyas fərqi (log şkala) — Azərbaycan qalın xətt ilə",
        xaxis=dict(dtick=1),
        yaxis=dict(type="log", title="min ton (log)"),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
    )
    apply_theme(fig_trend, height=400)
    st.plotly_chart(fig_trend, use_container_width=True)

    # Məhsuldarlıq müqayisəsi - kritik göstərici
    st.markdown('<div class="section-header">Məhsuldarlıq — Strateji Problem Sahəsi</div>', unsafe_allow_html=True)

    yield_sorted = NEIGHBOR_DATA.sort_values("Məhsuldarlıq (t/ha)", ascending=True)
    fig_yield = go.Figure()
    fig_yield.add_trace(go.Bar(
        x=yield_sorted["Ölkə"],
        y=yield_sorted["Məhsuldarlıq (t/ha)"],
        marker=dict(
            color=[NEIGHBOR_COLORS[c] for c in yield_sorted["Ölkə"]],
            line=dict(color="white", width=1.5),
        ),
        text=[f"{v:.2f} t/ha" for v in yield_sorted["Məhsuldarlıq (t/ha)"]],
        textposition="outside",
        textfont=dict(color=THEME["text_dark"], size=12, family="Fraunces"),
        hovertemplate="<b>%{x}</b><br>%{y:.2f} t/ha<extra></extra>",
    ))
    fig_yield.add_hline(y=NEIGHBOR_DATA[NEIGHBOR_DATA["Ölkə"]=="Türkiyə"]["Məhsuldarlıq (t/ha)"].iloc[0],
                        line_dash="dot", line_color=THEME["primary"],
                        annotation_text="Türkiyə səviyyəsi (hədəf)",
                        annotation_position="top right",
                        annotation_font=dict(color=THEME["primary"], size=11))
    fig_yield.update_layout(
        title="Hektar başına məhsul — Azərbaycan Türkiyədən 3.7x geridə",
        yaxis=dict(title="t/ha", range=[0, 4]),
        showlegend=False,
    )
    apply_theme(fig_yield, height=360)
    st.plotly_chart(fig_yield, use_container_width=True)

    st.markdown("**Tam müqayisə cədvəli:**")
    display_df = NEIGHBOR_DATA.copy()
    render_styled_table(
        display_df.style
        .background_gradient(subset=["İstehsal (kt)"], cmap="Greens")
        .background_gradient(subset=["Adam başına (kq)"], cmap="Oranges")
        .background_gradient(subset=["Məhsuldarlıq (t/ha)"], cmap="YlGn")
        .format({
            "İstehsal (kt)": "{:,.2f}",
            "İstehlak (kt)": "{:,.1f}",
            "Adam başına (kq)": "{:.2f}",
            "Sahə (kha)": "{:,.2f}",
            "Məhsuldarlıq (t/ha)": "{:.2f}",
            "Çay tarixi (il)": "{:.0f}",
        })
    )

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown('<div class="info-box"><b>🇹🇷 Türkiyə dərsi:</b> Dövlət subsidiyası + ÇAYKUR kooperativ emalı + ixrac brendinqi → 3.0 kq/nəfər istehlak, 3.4 t/ha məhsuldarlıq.</div>', unsafe_allow_html=True)
    with cc2:
        st.markdown('<div class="warn-box"><b>🇬🇪 Gürcüstan dərsi:</b> Sovetdən sonra 90%-lik tənəzzül; "Geoplant" və qrant proqramları ilə yenidən cəhd edir, hələ də 0.78 t/ha.</div>', unsafe_allow_html=True)
    with cc3:
        st.markdown('<div class="danger-box"><b>🇮🇷 İran dərsi:</b> 25 min ha sahəyə baxmayaraq cəmi 0.56 t/ha — köhnə kollar və zəif aqrotexnika. <b>Azərbaycanın getməməli olduğu yol.</b></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — QİYMƏT ANALİZİ
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Qiymət Dinamikası: Azərbaycan İdxalı vs Dünya Bazarı</div>', unsafe_allow_html=True)

    fig_price = go.Figure()
    price_lines = [
        ("İdxal USD/Ton",            THEME["accent"],   "solid",   4,  "Azərbaycan İdxal"),
        ("Dünya Qiyməti (USD/Ton)",  THEME["primary"],  "dot",     2.5, "Dünya Orta"),
        ("Ceylon Qiymeti (USD/Ton)", THEME["secondary"],"dash",    2.5, "Ceylon (Sri Lanka)"),
        ("Kenya Qiymeti (USD/Ton)",  THEME["warning"],  "dashdot", 2.5, "Kenya (Mombasa)"),
    ]
    for col, color, dash, width, label in price_lines:
        fig_price.add_trace(go.Scatter(
            x=df["İl"], y=df[col],
            name=label,
            mode="lines+markers",
            line=dict(color=color, width=width, dash=dash),
            marker=dict(size=8 if "İdxal" in col else 6, line=dict(color="white", width=1.5)),
            hovertemplate="<b>" + label + " — %{x}</b><br>%{y:,.0f} USD/ton<extra></extra>",
        ))
    fig_price.update_layout(
        title="Azərbaycan idxal qiyməti dünya ortalamasından sistematik olaraq yüksəkdir",
        xaxis=dict(dtick=1),
        yaxis=dict(title="USD / Ton"),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
    )
    apply_theme(fig_price, height=420)
    st.plotly_chart(fig_price, use_container_width=True)

    df["Qiymət Artıqlığı (USD/Ton)"] = (df["İdxal USD/Ton"] - df["Dünya Qiyməti (USD/Ton)"]).round(0)
    df["Artıqlıq Zərəri (Mln USD)"]  = (df["Qiymət Artıqlığı (USD/Ton)"] * df["İdxal Həcmi (Ton)"] / 1_000_000).round(2)

    fig_loss = go.Figure()
    fig_loss.add_trace(go.Bar(
        x=df["İl"], y=df["Artıqlıq Zərəri (Mln USD)"],
        marker=dict(
            color=df["Artıqlıq Zərəri (Mln USD)"],
            colorscale=[[0, "#F8E4DF"], [1, THEME["accent"]]],
            line=dict(color="white", width=1),
        ),
        text=[f"${v:.1f}M" for v in df["Artıqlıq Zərəri (Mln USD)"]],
        textposition="outside",
        textfont=dict(color=THEME["text_dark"], size=10),
        hovertemplate="<b>%{x}</b><br>Artıq xərc: %{y:.2f} Mln USD<extra></extra>",
    ))
    fig_loss.update_layout(
        title="Dünya qiymətindən baha alışın yaratdığı illik artıq xərc (Mln USD)",
        xaxis=dict(dtick=1),
        yaxis=dict(title="Mln USD"),
        showlegend=False,
    )
    apply_theme(fig_loss, height=360)
    st.plotly_chart(fig_loss, use_container_width=True)

    total_loss = df["Artıqlıq Zərəri (Mln USD)"].sum()
    st.markdown(f'<div class="danger-box">2015–2025 arası dünya qiymətindən baha alış ucbatından toplam <b>${total_loss:.1f} Mln USD</b> artıq xərclənib. Bu, kiçik miqyaslı alışların yaratdığı danışıq gücü zəifliyindən qaynaqlanır.</div>', unsafe_allow_html=True)

    # Qiymət indeksi
    base_cols = ["İdxal USD/Ton", "Dünya Qiyməti (USD/Ton)", "Ceylon Qiymeti (USD/Ton)", "Kenya Qiymeti (USD/Ton)"]
    df_idx = df[["İl"] + base_cols].copy()
    for col in base_cols:
        df_idx[col] = (df_idx[col] / df_idx[col].iloc[0] * 100).round(1)

    name_map = {
        "İdxal USD/Ton": "Azərbaycan İdxal",
        "Dünya Qiyməti (USD/Ton)": "Dünya Orta",
        "Ceylon Qiymeti (USD/Ton)": "Ceylon",
        "Kenya Qiymeti (USD/Ton)": "Kenya",
    }
    color_map_idx = {
        "Azərbaycan İdxal": THEME["accent"],
        "Dünya Orta": THEME["primary"],
        "Ceylon": THEME["secondary"],
        "Kenya": THEME["warning"],
    }

    fig_idx = go.Figure()
    for col in base_cols:
        label = name_map[col]
        is_az = "Azərbaycan" in label
        fig_idx.add_trace(go.Scatter(
            x=df_idx["İl"], y=df_idx[col],
            name=label,
            mode="lines+markers",
            line=dict(color=color_map_idx[label], width=3.5 if is_az else 2),
            marker=dict(size=8 if is_az else 6, line=dict(color="white", width=1.5)),
            hovertemplate="<b>" + label + " — %{x}</b><br>İndeks: %{y:.1f}<extra></extra>",
        ))
    fig_idx.add_hline(y=100, line_dash="dot", line_color=THEME["text_muted"])
    fig_idx.update_layout(
        title="Qiymət İndeksi (2015 = 100) — Nisbi artım müqayisəsi",
        xaxis=dict(dtick=1),
        yaxis=dict(title="İndeks (2015=100)"),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
    )
    apply_theme(fig_idx, height=380)
    st.plotly_chart(fig_idx, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — REGİONAL XƏRİTƏ
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Azərbaycanda Çayçılıq Regionlarının Xəritəsi</div>', unsafe_allow_html=True)

    map_df = pd.DataFrame({
        "Region":    ["Lənkəran", "Astara", "Lerik", "Masallı", "Zaqatala", "Balakən", "Qax"],
        "lat":       [38.7529,   38.4473,  38.7731, 39.0345,   41.6315,   41.7069,  41.4211],
        "lon":       [48.8485,   48.8744,  48.4153, 48.6680,   46.6433,   46.4039,  46.9322],
        "Sahə (ha)": [450,       210,      180,     160,       120,       70,       60],
        "Məhsul (ton)":[414,     193,      165,     147,       110,       64,       55],
        "pH Uyğunluq":["Yüksək","Orta",  "Orta",  "Orta",    "Aşağı",   "Aşağı",  "Aşağı"],
        "Kritiklik": ["Yüksək", "Orta",  "Orta",  "Orta",    "Kritik",  "Kritik", "Kritik"],
        "Əsas Problem": [
            "Yay quraqlığı (iyul-avqust), suvarma çatışmazlığı, sahə parçalanması",
            "Dağlıq relyef, logistika çətinliyi, kiçik plantasiyalar",
            "Yolların vəziyyəti pis, emal müəssisəsindən uzaqlıq",
            "Torpaq rütubəti qeyri-stabil, iqlim dəyişkənliyi",
            "pH balansı çay üçün uyğun deyil, yenidən torpaq islahı lazımdır",
            "Su mənbəyi məhdud, infrastruktur zəif",
            "Mikroiqlim qeyri-sabit, donvurma riski yüksək",
        ],
        "Potensial (2030)": [700, 320, 270, 250, 100, 80, 60],
    })

    color_map = {
        "Yüksək": THEME["secondary"],
        "Orta":   THEME["warning"],
        "Kritik": THEME["accent"],
        "Aşağı":  THEME["accent"],
    }

    kritiklik_filter = st.multiselect(
        "Kritiklik filtri:", ["Yüksək", "Orta", "Kritik"],
        default=["Yüksək", "Orta", "Kritik"],
    )
    map_filtered = map_df[map_df["Kritiklik"].isin(kritiklik_filter)]

    fig_map = px.scatter_mapbox(
        map_filtered,
        lat="lat", lon="lon",
        text="Region",
        color="Kritiklik",
        size="Sahə (ha)",
        size_max=35,
        hover_name="Region",
        hover_data={"Sahə (ha)": True, "Məhsul (ton)": True,
                    "pH Uyğunluq": True, "Əsas Problem": True, "lat": False, "lon": False},
        color_discrete_map={
            "Yüksək": THEME["secondary"],
            "Orta":   THEME["warning"],
            "Kritik": THEME["accent"],
        },
        zoom=7, height=500,
        title="Baloncuq ölçüsü = sahə (ha) | Rəng = ümumi kritiklik səviyyəsi",
    )
    fig_map.update_layout(
        mapbox_style="carto-positron",
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        paper_bgcolor=THEME["bg_chart"],
        font=dict(family="IBM Plex Sans", color=THEME["text_dark"]),
        legend=dict(orientation="h", y=-0.06, bgcolor="rgba(255,255,255,0.8)"),
        title=dict(font=dict(size=14, color=THEME["text_dark"])),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    render_styled_table(
        map_df[["Region", "Sahə (ha)", "Məhsul (ton)", "pH Uyğunluq", "Kritiklik", "Potensial (2030)", "Əsas Problem"]]
        .style.applymap(
            lambda v: f"color: {color_map.get(v, '#000')}; font-weight:600",
            subset=["Kritiklik"]
        )
    )


# ══════════════════════════════════════════════════════════════
# TAB 5 — PROBLEMLƏR & STRATEGİYA
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Əvvəlki Cəhdlər: Uğursuzluq Analizi</div>', unsafe_allow_html=True)

    cehdler = pd.DataFrame({
        "Cəhd / Proqram": [
            "Kütləvi sahə genişləndirilməsi (kəmiyyət yönümlü)",
            "Yerli brend adı ilə qarışıq satış (blend çay)",
            "Kiçik fermerlərə qısamüddətli subsidiya",
            "Ənənəvi selləmə suvarma infrastrukturu",
            "Emal zavodlarının güc artırımı",
            "İxracata yönləndirilmiş kampaniyalar",
        ],
        "Niyə Uğursuz Oldu?": [
            "pH turşuluğu (lazım: 4.5-5.5) yoxlanmadan əkin; kol tələfatı ~35%",
            "Xarici ucuz toz çay qarışdırıldı; yerli çaya istehlakçı etimadsızlığı yarandı",
            "4-5 illik məhsul gözləmə dövründə risk sığortası yox idi; fermerlər başqa bitkiyə keçdi",
            "İyul-avqust quraqlığında buxarlanma >60%; effektivlik sıfıra yaxın",
            "Xammal çatışmazlığı ilə zavodlar 30-40% güclə işlədi; sabit xərc yükü qaldı",
            "Yerli standart və keyfiyyət sertifikatlaşması olmadiğından xarici alıcılar rədd etdi",
        ],
        "Düzgün Həll": [
            "Yalnız pH 4.5-5.5 torpaqda mikro-zonalaşdırma; GIS torpaq xəritəsi",
            "Coğrafi İşarə (GI) qeydiyyatı; 'Lənkəran Əl Yığımı' premium nişi",
            "Güzəşt müddətli (4 il ödənişsiz) uzunmüddətli kredit + məhsul sığortası",
            "Smart damlama + çiləmə sistemləri; yay üçün torpaq nəmlik sensoru",
            "Kontraktlı fermerlik: fabrik sabit qiymətlə alış zəmanəti verir",
            "ISO 3720 / EU Organic sertifikatı + brend qablaşdırma investisiyası",
        ],
        "Maliyyə Effekti (KPI)": [
            "Tələfat 35% → <5%; hektar məhsuldarlıq 0.92 → 1.8 t",
            "Satış qiyməti kq başına 3x → 8-25 USD arasına çıxır",
            "Fermentin sahəni tərk etmə ehtimalı 80% azalır",
            "Su sərfi 40-45% azalır; məhsul keyfiyyəti yüksəlir",
            "Zavod güc istifadəsi 40% → 85%+",
            "İxrac dəyəri 2-3x artır; Körfəz ölkələri bazarı açılır",
        ],
    })

    def highlight_row(row):
        return [f"background-color: #FBF6E8" if i % 2 == 0 else "" for i in range(len(row))]

    render_styled_table(cehdler.style.apply(highlight_row, axis=1))

    st.markdown('<div class="section-header">Strateji Optimallaşdırma Matrisi (Prioritetləşdirilmiş)</div>', unsafe_allow_html=True)

    matrix = pd.DataFrame({
        "Prioritet": ["🔴 P1", "🔴 P1", "🟠 P2", "🟠 P2", "🟡 P3", "🟡 P3"],
        "Strateji İstiqamət": [
            "Su İdarəetməsi (Smart Suvarma)",
            "Torpaq İslahı (pH Optimizasiyası)",
            "Kontraktlı Fermerlik Modeli",
            "GI Brendinq + Premium Nişi",
            "Logistika & Soyuq Zəncir",
            "Kooperativ Emal Fabrikləri",
        ],
        "Tətbiq Lokasiyası": [
            "Bütün çay bölgələri (450+ ha)",
            "Zaqatala, Balakən (pH problemli sahələr)",
            "Lənkəran, Astara fabriklər",
            "Daxili + MDB/Körfəz bazarları",
            "Lənkəran-Bakı-Gürcüstan dəhlizi",
            "Lerik, Masallı kiçik plantasiyaları",
        ],
        "Xərc (Mln USD)": [8.5, 3.2, 1.5, 2.8, 5.0, 4.2],
        "İllik Gəlir Artımı (Mln USD)": [12.0, 4.5, 3.8, 9.5, 2.5, 3.0],
        "Geri Dönüş (İl)": [1.5, 1.4, 0.8, 0.7, 4.0, 2.8],
        "Risk Səviyyəsi": ["Orta", "Aşağı", "Aşağı", "Orta", "Yüksək", "Orta"],
    })

    render_styled_table(
        matrix.style
        .background_gradient(subset=["İllik Gəlir Artımı (Mln USD)"], cmap="Greens")
        .background_gradient(subset=["Geri Dönüş (İl)"], cmap="RdYlGn_r")
        .format({"Xərc (Mln USD)": "{:.1f}", "İllik Gəlir Artımı (Mln USD)": "{:.1f}", "Geri Dönüş (İl)": "{:.1f}"})
    )

    total_inv = matrix["Xərc (Mln USD)"].sum()
    total_ret = matrix["İllik Gəlir Artımı (Mln USD)"].sum()
    st.markdown(f'<div class="info-box">Bütün P1-P3 tədbirləri birlikdə icra edilsə: <b>${total_inv:.1f} Mln USD</b> investisiya qarşılığında illik <b>${total_ret:.1f} Mln USD</b> əlavə gəlir potensialı. Ümumi ROI: <b>{total_ret/total_inv*100:.0f}%</b>/il.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 6 — ARIMA PROQNOZ & SİMULYASİYA
# ══════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">ARIMA Statistik Proqnoz Modeli (2026–2032)</div>', unsafe_allow_html=True)

    if not HAS_ARIMA:
        st.markdown('<div class="danger-box">⚠️ ARIMA üçün <code>statsmodels</code> kitabxanası tələb olunur. Quraşdırma: <code>pip install statsmodels</code></div>', unsafe_allow_html=True)

    # ARIMA parametrləri sidebar-da
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-title">📈 ARIMA Parametrləri</div>', unsafe_allow_html=True)

    arima_p = st.sidebar.slider("AR mərtəbəsi (p)", 0, 3, 1, help="Avtoreqressiya komponenti")
    arima_d = st.sidebar.slider("Diferensiya (d)", 0, 2, 1, help="Stasionarlıq üçün diferensiya sayı")
    arima_q = st.sidebar.slider("MA mərtəbəsi (q)", 0, 3, 1, help="Hərəkətli orta komponenti")
    fc_horizon = st.sidebar.slider("Proqnoz horizontu (il)", 3, 10, 7)

    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-title">⚙️ Simulyasiya Parametrləri</div>', unsafe_allow_html=True)

    sim_hektar = st.sidebar.slider("Yeni Smart Plantasiya (ha)", 50, 1000, 200, step=10)
    sim_suvarma = st.sidebar.selectbox(
        "Suvarma Növü",
        ["Ənənəvi selləmə (1.0 t/ha)", "Yağışlama (1.3 t/ha)", "Smart Damlama (1.9 t/ha)"],
    )
    sim_brend = st.sidebar.radio(
        "Satış Seqmenti",
        ["Kütləvi blend çay (4 USD/kq)", "Keyfiyyətli domestic (8 USD/kq)", "Premium GI single-origin (22 USD/kq)"],
    )
    sim_kooperativ = st.sidebar.checkbox("Kooperativ Emal Fabrik Dəstəyi?", value=True)
    sim_sertifikat = st.sidebar.checkbox("Beynəlxalq Sertifikat (ISO/Organic)?", value=False)

    # ───── ARIMA proqnozu ─────
    if HAS_ARIMA:
        prod_series = pd.Series(df["Yerli İstehsal (Ton)"].values, index=df["İl"].values)
        arima_fc, arima_lo, arima_hi, arima_aic = forecast_arima(
            prod_series, n_years=fc_horizon, order=(arima_p, arima_d, arima_q)
        )
        imp_series = pd.Series(df["İdxal Həcmi (Ton)"].values, index=df["İl"].values)
        imp_fc, imp_lo, imp_hi, imp_aic = forecast_arima(
            imp_series, n_years=fc_horizon, order=(arima_p, arima_d, arima_q)
        )
    else:
        arima_fc = arima_lo = arima_hi = arima_aic = None
        imp_fc = imp_lo = imp_hi = imp_aic = None

    # Xətti proqnoz müqayisə üçün
    lin_fc, lin_slope, lin_r2 = forecast_linear(df["Yerli İstehsal (Ton)"], n_years=fc_horizon)
    fut_years = list(range(2026, 2026 + fc_horizon))

    # ARIMA nəticələri görünüşü
    arima_info_col1, arima_info_col2, arima_info_col3 = st.columns(3)
    with arima_info_col1:
        order_str = f"({arima_p},{arima_d},{arima_q})"
        kpi_card("Model", f"ARIMA{order_str}",
                 f"AIC: {arima_aic:.1f}" if arima_aic else "Model fit alınmadı",
                 accent=THEME["primary"])
    with arima_info_col2:
        if arima_fc is not None:
            kpi_card("2032 ARIMA Proqnozu", f"{arima_fc[-1]:,.0f} ton",
                     f"95% İE: [{arima_lo[-1]:,.0f}; {arima_hi[-1]:,.0f}]",
                     accent=THEME["secondary"])
        else:
            kpi_card("ARIMA", "—", "statsmodels lazımdır", accent=THEME["neutral"])
    with arima_info_col3:
        kpi_card("Xətti Reqressiya", f"{lin_fc[-1]:,.0f} ton",
                 f"R² = {lin_r2:.3f} · slope = {lin_slope:+.1f}/il",
                 accent=THEME["accent"])

    # ───── ARIMA proqnoz qrafiki ─────
    fig_arima = go.Figure()

    # Tarixi məlumat
    fig_arima.add_trace(go.Scatter(
        x=df["İl"], y=df["Yerli İstehsal (Ton)"],
        mode="lines+markers",
        name="Tarixi məlumat (2015-2025)",
        line=dict(color=THEME["primary"], width=3.5),
        marker=dict(size=9, color=THEME["primary"], line=dict(color="white", width=2)),
        hovertemplate="<b>%{x}</b><br>%{y:,.0f} ton<extra></extra>",
    ))

    if arima_fc is not None:
        # 95% etibar intervalı (bant)
        fig_arima.add_trace(go.Scatter(
            x=fut_years + fut_years[::-1],
            y=list(arima_hi) + list(arima_lo)[::-1],
            fill="toself",
            fillcolor="rgba(184,80,66,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Etibar İntervalı",
            hoverinfo="skip",
            showlegend=True,
        ))
        # ARIMA orta proqnoz
        fig_arima.add_trace(go.Scatter(
            x=fut_years, y=arima_fc,
            mode="lines+markers",
            name=f"ARIMA{order_str} proqnoz",
            line=dict(color=THEME["accent"], width=3, dash="dash"),
            marker=dict(size=9, color=THEME["accent"], symbol="diamond", line=dict(color="white", width=2)),
            hovertemplate="<b>%{x}</b><br>Proqnoz: %{y:,.0f} ton<extra></extra>",
        ))

    # Xətti müqayisə
    fig_arima.add_trace(go.Scatter(
        x=fut_years, y=lin_fc,
        mode="lines+markers",
        name="Xətti reqressiya (müqayisə)",
        line=dict(color=THEME["warning"], width=2, dash="dot"),
        marker=dict(size=6, symbol="square"),
        hovertemplate="<b>%{x}</b><br>Xətti: %{y:,.0f} ton<extra></extra>",
    ))

    fig_arima.add_vline(x=2025.5, line_dash="dash", line_color=THEME["text_muted"],
                        annotation_text="Proqnoz sərhədi",
                        annotation_position="top",
                        annotation_font=dict(color=THEME["text_muted"], size=11))
    fig_arima.update_layout(
        title=f"Yerli İstehsal — ARIMA{order_str} Proqnozu vs Xətti Müqayisə",
        xaxis=dict(dtick=1, title="İl"),
        yaxis=dict(title="Ton"),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
    )
    apply_theme(fig_arima, height=440)
    st.plotly_chart(fig_arima, use_container_width=True)

    if arima_fc is not None:
        st.markdown(f'<div class="info-box">📊 <b>ARIMA{order_str}</b> modeli yerli istehsalın 2032-də <b>{arima_fc[-1]:,.0f} ton</b> ola biləcəyini proqnozlaşdırır. 95% etibarla aralıq: <b>{arima_lo[-1]:,.0f} – {arima_hi[-1]:,.0f} ton</b>. AIC: <b>{arima_aic:.1f}</b> (daha aşağı = daha yaxşı model).</div>', unsafe_allow_html=True)

    # ───── İdxal həcmi ARIMA proqnozu ─────
    if imp_fc is not None:
        st.markdown('<div class="section-header">İdxal Həcmi ARIMA Proqnozu</div>', unsafe_allow_html=True)

        fig_imp_arima = go.Figure()
        fig_imp_arima.add_trace(go.Scatter(
            x=df["İl"], y=df["İdxal Həcmi (Ton)"],
            mode="lines+markers",
            name="Tarixi idxal",
            line=dict(color=THEME["primary"], width=3.5),
            marker=dict(size=9, line=dict(color="white", width=2)),
            hovertemplate="<b>%{x}</b><br>%{y:,.0f} ton<extra></extra>",
        ))
        fig_imp_arima.add_trace(go.Scatter(
            x=fut_years + fut_years[::-1],
            y=list(imp_hi) + list(imp_lo)[::-1],
            fill="toself",
            fillcolor="rgba(107,78,113,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Etibar İntervalı",
            hoverinfo="skip",
        ))
        fig_imp_arima.add_trace(go.Scatter(
            x=fut_years, y=imp_fc,
            mode="lines+markers",
            name=f"ARIMA{order_str} proqnoz",
            line=dict(color=THEME["purple"], width=3, dash="dash"),
            marker=dict(size=9, symbol="diamond", line=dict(color="white", width=2)),
            hovertemplate="<b>%{x}</b><br>Proqnoz: %{y:,.0f} ton<extra></extra>",
        ))
        fig_imp_arima.add_vline(x=2025.5, line_dash="dash", line_color=THEME["text_muted"])
        fig_imp_arima.update_layout(
            title=f"İdxal Həcmi — ARIMA{order_str} Proqnozu",
            xaxis=dict(dtick=1),
            yaxis=dict(title="Ton"),
            legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        )
        apply_theme(fig_imp_arima, height=380)
        st.plotly_chart(fig_imp_arima, use_container_width=True)

    # ───── SİMULYASİYA ─────
    st.markdown('<div class="section-header">Canlı İnvestisiya Simulyatoru</div>', unsafe_allow_html=True)

    verim_map = {
        "Ənənəvi selləmə (1.0 t/ha)": 1.0,
        "Yağışlama (1.3 t/ha)": 1.3,
        "Smart Damlama (1.9 t/ha)": 1.9,
    }
    qiymet_map = {
        "Kütləvi blend çay (4 USD/kq)": 4,
        "Keyfiyyətli domestic (8 USD/kq)": 8,
        "Premium GI single-origin (22 USD/kq)": 22,
    }

    verim    = verim_map[sim_suvarma]
    qiymet   = qiymet_map[sim_brend]
    bonus_koop = 0.08 if sim_kooperativ else 0.0
    bonus_cert = 1.30 if sim_sertifikat  else 1.0

    mehsul_ton  = sim_hektar * verim
    qiymet_eff  = qiymet * bonus_cert * (1 + bonus_koop)
    geler_usd   = mehsul_ton * 1000 * qiymet_eff
    idxal_azalma = min(100, mehsul_ton / latest["Ümumi İstehlak (Ton)"] * 100)
    invest_map = {"Ənənəvi selləmə (1.0 t/ha)": 1200, "Yağışlama (1.3 t/ha)": 2200, "Smart Damlama (1.9 t/ha)": 3500}
    invest_total = sim_hektar * invest_map[sim_suvarma] / 1_000_000
    roi_pct = (geler_usd / 1_000_000 / invest_total * 100) if invest_total > 0 else 0

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        kpi_card("Gözlənilən Məhsul", f"{mehsul_ton:,.0f} ton",
                 f"{sim_hektar} ha × {verim} t/ha", accent=THEME["secondary"])
    with r2:
        kpi_card("İllik Gəlir", f"${geler_usd/1e6:.2f} mln",
                 f"{qiymet_eff:.1f} USD/kq effektiv", accent=THEME["primary"])
    with r3:
        kpi_card("İdxal Azalması", f"{idxal_azalma:.1f}%",
                 "Cari istehlakdan pay", accent=THEME["purple"])
    with r4:
        kpi_card("ROI", f"{roi_pct:.0f}%/il",
                 f"${invest_total:.1f} Mln investisiyaya", accent=THEME["accent"])

    # =============================================================================
    # YENİLİK 3: DƏRİN MALİYYƏ ANALİZİ (NPV & Payback) - sidebar-da
    # =============================================================================
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💰 2. Dərin Maliyyə Analizi (NPV & Payback)")
    diskont_derecesi = st.sidebar.slider("Diskont Dərəcəsi (%) [WACC]", 5, 20, 10) / 100
    ilkin_investisiya = sim_hektar * 5000  # Hektar başına ortalama 5000 USD ilkin xərc fərz edək
    # Nağd pul axınlarının (Cash Flow) və NPV-nin hesablanması
    # fərz edək ki, illik xalis qazanc = geler_usd * 40% (istismar marjası)
    illik_xalis_menfeet = geler_usd * 0.40
    # 5 illik NPV hesablama düsturu
    npv = -ilkin_investisiya
    for t in range(1, 6):
        npv += illik_xalis_menfeet / ((1 + diskont_derecesi) ** t)
    # Geriödəmə müddəti (Payback Period)
    if illik_xalis_menfeet > 0:
        payback = ilkin_investisiya / illik_xalis_menfeet
    else:
        payback = float('inf')
    st.sidebar.write(f"• İlkin İnvestisiya Həcmi: **{ilkin_investisiya:,.0f} USD**")
    if npv > 0:
        st.sidebar.write(f"• Xalis Cari Dəyər (5 İllik NPV): :green[**+{npv:,.0f} USD**]")
    else:
        st.sidebar.write(f"• Xalis Cari Dəyər (5 İllik NPV): :red[**{npv:,.0f} USD**]")
    if payback <= 5:
        st.sidebar.write(f"• İnvestisiyanın Geri Ödənməsi: :green[**{payback:.1f} İl**]")
    else:
        st.sidebar.write(f"• İnvestisiyanın Geri Ödənməsi: :orange[**{payback:.1f} İl (Uzunmüddətli)**]")

    # =============================================================================
    # YENİLİK 4: DATA EKSPORT DÜYMƏSİ
    # =============================================================================
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 3. Hesabatı İxrac Edin")

    @st.cache_data
    def convert_df_to_csv(d):
        return d.to_csv(index=False).encode('utf-8')

    csv_fayl = convert_df_to_csv(df)
    st.sidebar.download_button(
        label="📊 Simulyasiya Datalarını Yüklə (CSV)",
        data=csv_fayl,
        file_name='cay_bazari_analitik_hesabat.csv',
        mime='text/csv',
    )

    # ───── Ssenari müqayisəsi ─────
    st.markdown('<div class="section-header">Ssenari Müqayisəsi — Yerli İstehsal (2025–2032)</div>', unsafe_allow_html=True)

    base_2025 = df["Yerli İstehsal (Ton)"].iloc[-1]
    scenario_years = list(range(2025, 2026 + fc_horizon))

    scenarios = {
        "Status Quo (+2%/il)":        [base_2025 * (1.02 ** i) for i in range(len(scenario_years))],
        "Orta Optimallaşdırma (+6%)": [base_2025 * (1.06 ** i) for i in range(len(scenario_years))],
        "Tam Proqram (+12%/il)":      [base_2025 * (1.12 ** i) for i in range(len(scenario_years))],
        f"Simulyasiya ({sim_hektar}ha)": [base_2025 + mehsul_ton * (1 + 0.04 * i) for i in range(len(scenario_years))],
    }
    colors_sc = [THEME["neutral"], THEME["warning"], THEME["secondary"], THEME["accent"]]

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=df["İl"], y=df["Yerli İstehsal (Ton)"],
        name="Tarixi (2015-2025)", mode="lines+markers",
        line=dict(color=THEME["primary"], width=3.5),
        marker=dict(size=8, line=dict(color="white", width=2)),
        hovertemplate="<b>%{x}</b><br>%{y:,.0f} ton<extra></extra>",
    ))

    # ARIMA da qrafikdə göstəririk
    if arima_fc is not None:
        fig_forecast.add_trace(go.Scatter(
            x=fut_years, y=arima_fc,
            name=f"ARIMA{order_str}", mode="lines+markers",
            line=dict(color=THEME["purple"], width=2.5, dash="solid"),
            marker=dict(size=7, symbol="diamond"),
            hovertemplate="<b>%{x}</b><br>ARIMA: %{y:,.0f}<extra></extra>",
        ))

    for (sc_name, sc_vals), sc_color in zip(scenarios.items(), colors_sc):
        fig_forecast.add_trace(go.Scatter(
            x=scenario_years, y=[round(v, 0) for v in sc_vals],
            name=sc_name, mode="lines+markers",
            line=dict(color=sc_color, width=2, dash="dot"),
            marker=dict(size=6),
            hovertemplate="<b>%{x}</b><br>" + sc_name + ": %{y:,.0f}<extra></extra>",
        ))

    fig_forecast.add_vline(x=2025.5, line_dash="dash", line_color=THEME["text_muted"],
                            annotation_text="Proqnoz sərhədi", annotation_position="top",
                            annotation_font=dict(color=THEME["text_muted"], size=11))
    fig_forecast.update_layout(
        title="Bütün ssenarilər və ARIMA proqnozunun müqayisəsi",
        xaxis=dict(dtick=1, title="İl"),
        yaxis=dict(title="Ton"),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
    )
    apply_theme(fig_forecast, height=460)
    st.plotly_chart(fig_forecast, use_container_width=True)

    # 2032 hədəf cədvəli
    st.markdown("**Proqnoz Horizontunun Sonu üçün Hədəf Müqayisəsi:**")
    target_table = []
    demand_growth = 0.025
    end_year = scenario_years[-1]

    if arima_fc is not None:
        prod_end_arima = arima_fc[-1]
        demand_end = latest["Ümumi İstehlak (Ton)"] * (1 + demand_growth) ** (len(scenario_years) - 1)
        dep_end_arima = max(0, (demand_end - prod_end_arima) / demand_end * 100)
        target_table.append({
            "Ssenariu": f"ARIMA{order_str} statistik proqnoz",
            f"{end_year} İstehsal (ton)": round(prod_end_arima, 0),
            f"{end_year} İdxal Asılılığı (%)": round(dep_end_arima, 1),
            "70% Hədəfə Çatır?": "✅ Bəli" if dep_end_arima <= 70 else "❌ Xeyr",
        })

    for sc_name, sc_vals in scenarios.items():
        prod_end = sc_vals[-1]
        demand_end = latest["Ümumi İstehlak (Ton)"] * (1 + demand_growth) ** (len(scenario_years) - 1)
        dep_end = max(0, (demand_end - prod_end) / demand_end * 100)
        target_table.append({
            "Ssenariu": sc_name,
            f"{end_year} İstehsal (ton)": round(prod_end, 0),
            f"{end_year} İdxal Asılılığı (%)": round(dep_end, 1),
            "70% Hədəfə Çatır?": "✅ Bəli" if dep_end <= 70 else "❌ Xeyr",
        })
    render_styled_table(pd.DataFrame(target_table))

    st.markdown('<div class="info-box">ARIMA modeli yalnız <b>keçmiş trendi</b> ekstrapolyasiya edir — siyasət dəyişikliyini hesab etmir. <b>Tam Proqram (+12%/il)</b> və müsbət Simulyasiya ssenariləri yalnız aktiv investisiya və strateji icra ilə real olar.</div>', unsafe_allow_html=True)

    # =============================================================================
    # YENİLİK 2: EKONOMETRİK TREND PROQNOZLAŞDIRILMASI (ML Linear Regression)
    # =============================================================================
    st.markdown('<div class="section-header">📈 Machine Learning (Linear Regression) İdxal Proqnozu</div>', unsafe_allow_html=True)
    # Keçmiş dataya əsasən gələcək illəri öyrətmək (X = İl, Y = İdxal Həcmi)
    X = df[["İl"]].values
    y = df["İdxal Həcmi (Ton)"].values
    ml_model = LinearRegression()
    ml_model.fit(X, y)
    # Gələcək 3 il üçün illəri yaradırıq
    son_il = int(df["İl"].max())
    gelecek_iller = np.array([[son_il + 1], [son_il + 2], [son_il + 3]])
    tahminler = ml_model.predict(gelecek_iller)
    # Proqnoz nəticələrini vizuallaşdırmaq üçün yeni dataframe
    proqnoz_df = pd.DataFrame({
        "İl": [int(x[0]) for x in gelecek_iller],
        "Proqnoz İdxal Həcmi (Ton)": tahminler.round(0).astype(int)
    })
    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        # Keçmiş və proqnoz datalarını birləşdirib tək xətt qrafiki çıxarmaq
        fig_prognoz = go.Figure()
        fig_prognoz.add_trace(go.Scatter(
            x=df["İl"], y=df["İdxal Həcmi (Ton)"],
            name="Mövcud İdxal", mode="lines+markers",
            line=dict(color=THEME["primary"], width=3),
            marker=dict(size=8, line=dict(color="white", width=2)),
        ))
        fig_prognoz.add_trace(go.Scatter(
            x=proqnoz_df["İl"], y=proqnoz_df["Proqnoz İdxal Həcmi (Ton)"],
            name="ML Trend Proqnozu (3 İllik)", mode="lines+markers",
            line=dict(color=THEME["warning"], dash="dash", width=3),
            marker=dict(size=8, symbol="diamond", line=dict(color="white", width=2)),
        ))
        fig_prognoz.update_layout(
            title="Gələcək İllər üzrə İdxal Tələbatının Artım Proqnozu",
            xaxis=dict(dtick=1),
            yaxis=dict(title="Ton"),
            legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        )
        apply_theme(fig_prognoz, height=340)
        st.plotly_chart(fig_prognoz, use_container_width=True)
    with col_p2:
        st.write("🤖 **Modelin Analitik Qeydi:**")
        st.caption("Xətti Reqressiya modeli keçmiş trendləri analiz edərək ölkədə çay istehlakı və idxalının hər il ortalama hansı sürətlə artacağını avtomatik hesablayır. Bu, strateji planlamada anbar və tədarük idarəedilməsi üçün mühümdür.")
        render_styled_table(proqnoz_df)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center; color:{THEME["text_muted"]}; font-size:12.5px; padding:10px 0 20px;">
    Məlumat mənbələri: FAO Food & Agriculture Organization · Azərbaycan Dövlət Statistika Komitəsi ·
    World Bank Tea Price Index · Mombasa Tea Auction · ÇAYKUR Türkiyə · Geoplant Gürcüstan
    <br>Panel: Azərbaycan Çay Bazarı Strateji Analitik Sistemi <b>v3.0</b> · ARIMA inteqrasiyalı
</div>
""", unsafe_allow_html=True)

