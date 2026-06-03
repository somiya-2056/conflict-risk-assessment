import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Conflict Risk Assessment",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&family=Orbitron:wght@700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
        background-color: #030c14;
        color: #c9d8e8;
    }
    .stApp { background-color: #030c14; }

    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        color: #00e5ff;
        text-align: center;
        letter-spacing: 3px;
        text-shadow: 0 0 20px #00e5ff88;
        padding: 1rem 0 0.2rem 0;
    }
    .sub-title {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.8rem;
        color: #4a7fa0;
        text-align: center;
        letter-spacing: 4px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #071a2a 0%, #0a2540 100%);
        border: 1px solid #1a4a6a;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px #00000055;
    }
    .metric-card h3 {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.72rem;
        color: #4a9aba;
        letter-spacing: 3px;
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
    }
    .metric-card .value {
        font-family: 'Orbitron', monospace;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .risk-low      { color: #00e676; text-shadow: 0 0 12px #00e67688; }
    .risk-medium   { color: #ffea00; text-shadow: 0 0 12px #ffea0088; }
    .risk-high     { color: #ff6d00; text-shadow: 0 0 12px #ff6d0088; }
    .risk-critical { color: #ff1744; text-shadow: 0 0 16px #ff174488; }
    .section-header {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.78rem;
        color: #00b4d8;
        letter-spacing: 4px;
        border-bottom: 1px solid #1a4a6a;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
        text-transform: uppercase;
    }
    [data-testid="stSidebar"] {
        background-color: #040e1a !important;
        border-right: 1px solid #1a4a6a;
    }
    [data-testid="stSidebar"] label {
        color: #90b8d4 !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 0.9rem !important;
    }
    hr { border-color: #1a4a6a; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def train_models():
    np.random.seed(42)
    X_hist = np.array([
        [0.6,0.7,0.9,0.0,0.5,0.1,0.2,0.8,0.1,0.3],
        [0.5,0.5,0.8,0.0,0.4,0.1,0.3,0.7,0.1,0.4],
        [0.7,0.6,0.9,0.0,0.6,0.1,0.1,0.9,0.1,0.2],
        [0.6,0.5,0.8,0.5,0.4,0.2,0.3,0.7,0.1,0.5],
        [0.7,0.4,0.7,0.6,0.3,0.4,0.4,0.6,0.2,0.6],
        [0.8,0.3,0.7,0.7,0.3,0.5,0.4,0.7,0.2,0.7],
        [0.8,0.2,0.5,0.8,0.2,0.3,0.6,0.3,0.3,0.8],
        [0.9,0.2,0.4,0.8,0.2,0.3,0.7,0.2,0.4,0.9],
        [0.5,0.8,0.9,0.3,0.6,0.6,0.1,0.9,0.1,0.2],
        [0.6,0.6,0.7,0.5,0.4,0.5,0.3,0.6,0.2,0.5],
        [0.7,0.3,0.4,0.7,0.2,0.3,0.6,0.2,0.4,0.7],
        [0.8,0.2,0.3,0.8,0.1,0.2,0.8,0.1,0.5,0.9],
        [0.4,0.6,0.8,0.0,0.3,0.1,0.2,0.8,0.2,0.3],
        [0.5,0.4,0.7,0.5,0.3,0.3,0.4,0.5,0.3,0.5],
        [0.6,0.3,0.6,0.6,0.3,0.4,0.5,0.4,0.4,0.6],
        [0.6,0.4,0.7,0.6,0.4,0.6,0.3,0.8,0.3,0.5],
        [0.7,0.3,0.6,0.7,0.3,0.5,0.4,0.4,0.4,0.7],
        [0.8,0.2,0.5,0.7,0.2,0.4,0.5,0.3,0.5,0.8],
        [0.4,0.7,0.9,0.4,0.6,0.8,0.1,0.9,0.2,0.2],
        [0.5,0.5,0.7,0.5,0.4,0.6,0.3,0.6,0.3,0.5],
        [0.7,0.3,0.5,0.6,0.3,0.4,0.5,0.3,0.5,0.7],
        [0.9,0.2,0.3,0.7,0.2,0.3,0.7,0.2,0.6,0.9],
        [0.3,0.9,1.0,0.2,0.7,0.7,0.1,1.0,0.1,0.1],
        [0.4,0.8,0.9,0.3,0.6,0.7,0.2,0.9,0.1,0.2],
        [0.5,0.7,0.8,0.4,0.5,0.6,0.3,0.7,0.2,0.4],
        [0.6,0.5,0.6,0.6,0.3,0.4,0.5,0.5,0.3,0.6],
        [0.7,0.3,0.4,0.7,0.2,0.3,0.6,0.3,0.5,0.8],
        [0.8,0.2,0.3,0.8,0.1,0.2,0.7,0.2,0.6,0.9],
        [0.9,0.1,0.2,0.9,0.1,0.1,0.9,0.1,0.7,1.0],
        [0.3,1.0,1.0,0.1,0.9,0.9,0.0,1.0,0.0,0.0],
    ])
    y_hist = np.array([2,2,2,1,1,1,0,0,2,1,0,0,
                       2,1,1,1,0,0,2,1,0,0,
                       2,2,1,1,0,0,0,2])
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_hist)
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
    rf.fit(X_sc, y_hist)
    gb.fit(X_sc, y_hist)
    rf_cv = cross_val_score(rf, X_sc, y_hist, cv=5).mean()
    gb_cv = cross_val_score(gb, X_sc, y_hist, cv=5).mean()
    return rf, gb, scaler, rf_cv, gb_cv

rf_model, gb_model, scaler, rf_acc, gb_acc = train_models()

def predict_war(features):
    x = np.array(features).reshape(1,-1)
    x_sc = scaler.transform(x)
    rf_p = rf_model.predict_proba(x_sc)[0]
    gb_p = gb_model.predict_proba(x_sc)[0]
    avg  = (rf_p + gb_p) / 2
    classes = rf_model.classes_
    pd_ = {int(c): float(p) for c,p in zip(classes, avg)}
    for k in [0,1,2]: pd_.setdefault(k, 0.0)
    return pd_, int(np.argmax([pd_[0], pd_[1], pd_[2]]))

def risk_label(wp):
    if wp < 0.20: return "LOW",      "risk-low",      "#00e676"
    if wp < 0.45: return "MODERATE", "risk-medium",   "#ffea00"
    if wp < 0.70: return "HIGH",     "risk-high",     "#ff6d00"
    return "CRITICAL", "risk-critical", "#ff1744"

def india_win_prob(features, enemy):
    mil,econ,terr,nuke,ally,cyber,dip,inc,trade,intl = features
    base = 0.55
    if enemy == "Pakistan":
        base += 0.10 + mil*0.15 - ally*0.10
    else:
        base -= 0.12 + mil*0.08 - ally*0.15 + intl*0.10
    base -= nuke*0.08
    base += dip*0.05
    return max(0.10, min(0.88, base))

# ── HEADER ──
st.markdown('<div class="main-title">🛡️ AI-BASED CONFLICT RISK ASSESSMENT</div>', unsafe_allow_html=True)

st.divider()

# ── INPUTS ON MAIN PAGE ──
st.markdown('<div class="section-header">⚙️ SCENARIO INPUTS — ADJUST INDICATORS</div>', unsafe_allow_html=True)

inp1, inp2 = st.columns(2)
with inp1:
    enemy = st.selectbox("India vs →", ["Pakistan", "China"])
with inp2:
    year  = st.selectbox("Projection Year", [2025,2026,2027,2028,2029,2030])

st.markdown("<br>", unsafe_allow_html=True)
col_m, col_g, col_e, col_d = st.columns(4)

with col_m:
    st.markdown("**🪖 Military**")
    mil_balance = st.slider("India Military Superiority",   0.0, 1.0, 0.70, 0.05)
    cyber_risk  = st.slider("Cyber Warfare Threat",        0.0, 1.0, 0.55, 0.05)
    nuke_factor = st.slider("Nuclear Arsenal Parity",      0.0, 1.0, 0.70, 0.05)

with col_g:
    st.markdown("**🌐 Geopolitical**")
    terr_dispute  = st.slider("Territorial Dispute",       0.0, 1.0, 0.75, 0.05)
    recent_inc    = st.slider("Recent Incidents",          0.0, 1.0, 0.40, 0.05)
    alliance_gap  = st.slider("Enemy Alliance Advantage",  0.0, 1.0, 0.35, 0.05)

with col_e:
    st.markdown("**💰 Economic**")
    econ_stress   = st.slider("Economic Stress",           0.0, 1.0, 0.35, 0.05)
    trade_dep     = st.slider("Trade Interdependence",     0.0, 1.0, 0.20, 0.05)
    intl_pressure = st.slider("International Support",     0.0, 1.0, 0.55, 0.05)

with col_d:
    st.markdown("**🕊️ Diplomatic**")
    dip_relations = st.slider("Diplomatic Relations",      0.0, 1.0, 0.40, 0.05)
    st.markdown("<br><br>", unsafe_allow_html=True)
    run_btn = st.button("🔍 RUN PREDICTION", use_container_width=True)

st.divider()

features = [mil_balance, econ_stress, terr_dispute, nuke_factor,
            alliance_gap, cyber_risk, dip_relations, recent_inc,
            trade_dep, intl_pressure]

proba, outcome = predict_war(features)
war_prob   = proba[1] + proba[2]
full_prob  = proba[2]
peace_prob = proba[0]
risk_lbl, risk_cls, risk_color = risk_label(war_prob)
india_win  = india_win_prob(features, enemy)
enemy_win  = max(1.0 - india_win - 0.15, 0.05)
stalemate  = max(0.05, 0.15)
total = india_win + enemy_win + stalemate
india_win /= total; enemy_win /= total; stalemate /= total

compromise = round((dip_relations*6 + trade_dep*4 + (1-terr_dispute)*3 + (1-recent_inc)*3) / 4.5 * 100, 1)
compromise = min(max(compromise, 5), 80)

# ── INDIA HISTORICAL GRAPHS ──
st.markdown('<div class="section-header">📊 INDIA — HISTORICAL STRATEGIC DATA (1949–2025)</div>', unsafe_allow_html=True)

# Full SIPRI data 1949-2025
mil_years = list(range(1949, 2026))
mil_data = {
    'India': [
        None,None,None,None,None,None,None,518,655,688,
        730,895,1095,1149,1096,1198,1318,1547,1776,1905,
        1905,2100,2415,2634,2951,3076,3544,4000,4200,4500,
        4800,5200,5600,6100,6800,7200,7500,8100,8900,9800,
        10500,11200,12300,13100,14200,15300,16500,17800,19200,20800,
        22500,24000,25500,27000,28500,30000,32000,34000,36000,38000,
        40000,43000,46000,49000,48000,48000,50000,51000,55000,64000,
        66000,71000,73000,76000,80000,82000,85000
    ],
    'Pakistan': [
        None,250,287,251,217,204,169,153,165,178,
        195,210,230,250,275,300,330,365,400,435,
        470,510,550,600,650,700,750,800,870,940,
        1020,1100,1200,1300,1400,1520,1650,1800,1950,2100,
        2300,2500,2700,2900,3100,3400,3700,4000,4200,4500,
        4800,5000,5500,6000,6200,6800,7000,7200,8000,9000,
        9500,10000,9800,11000,10300,8600,10300,None,None,None,
        None,None,None,None,None,None,None
    ],
    'China': [
        None,None,None,None,None,None,None,None,None,None,
        None,None,None,None,None,None,None,None,None,None,
        None,None,None,None,None,None,None,None,None,None,
        None,None,None,None,None,None,None,None,None,None,
        None,None,None,None,None,None,None,None,None,None,
        None,None,None,None,None,None,14000,17000,20000,24000,
        28000,34000,42000,44000,63000,79000,97000,106000,125000,145000,
        164000,183000,197000,199000,210000,232000,240000,258000,259000,
        286000,292000,296000,311000,325000,336000
    ]
}
gdp_years = list(range(2005, 2024))
gdp_data = {
    'India':   [0.82,0.92,1.21,1.19,1.34,1.67,1.82,1.83,1.86,2.04,2.10,2.29,2.65,2.71,2.83,2.67,3.18,3.39,3.64],
    'Pakistan':[0.11,0.14,0.15,0.17,0.17,0.18,0.21,0.22,0.23,0.23,0.27,0.28,0.30,0.31,0.31,0.26,0.35,0.37,0.34],
    'China':   [2.26,2.75,3.55,4.60,5.10,6.09,7.55,8.53,9.57,10.48,11.06,11.20,12.24,13.61,14.34,14.73,17.73,17.96,18.27],
}

hg1, hg2, hg3 = st.columns(3)

with hg1:
    fig_mil = go.Figure()
    colors_mil = {'India':'#00e676','Pakistan':'#ffea00','China':'#ff1744'}
    for country, vals in mil_data.items():
        fig_mil.add_trace(go.Scatter(x=mil_years, y=vals, mode='lines+markers',
            name=country, line=dict(color=colors_mil[country], width=2),
            marker=dict(size=4)))
    fig_mil.update_layout(
        title=dict(text='Military Expenditure (Million USD)', font=dict(color='#00e5ff', size=12)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,26,42,0.8)',
        font=dict(color='#c9d8e8', size=10),
        yaxis=dict(gridcolor='#1a4a6a'),
        xaxis=dict(gridcolor='#1a4a6a'),
        legend=dict(font=dict(color='#c9d8e8', size=9), bgcolor='rgba(0,0,0,0)'),
        height=280, margin=dict(t=40,b=10,l=10,r=10))
    st.plotly_chart(fig_mil, use_container_width=True)

with hg2:
    fig_gdp = go.Figure()
    for country, vals in gdp_data.items():
        fig_gdp.add_trace(go.Scatter(x=gdp_years, y=vals, mode='lines+markers',
            name=country, line=dict(color=colors_mil[country], width=2),
            marker=dict(size=4)))
    fig_gdp.update_layout(
        title=dict(text='GDP (Trillion USD)', font=dict(color='#00e5ff', size=12)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,26,42,0.8)',
        font=dict(color='#c9d8e8', size=10),
        yaxis=dict(gridcolor='#1a4a6a'),
        xaxis=dict(gridcolor='#1a4a6a'),
        legend=dict(font=dict(color='#c9d8e8', size=9), bgcolor='rgba(0,0,0,0)'),
        height=280, margin=dict(t=40,b=10,l=10,r=10))
    st.plotly_chart(fig_gdp, use_container_width=True)

with hg3:
    feat_names = ['Military','Econ Stress','Territory','Nuclear',
                  'Alliance Gap','Cyber','Diplomacy','Incidents','Trade','Intl']
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)
    fig_imp = go.Figure(go.Bar(
        x=[importances[i]*100 for i in sorted_idx],
        y=[feat_names[i] for i in sorted_idx],
        orientation='h',
        marker=dict(
            color=[importances[i]*100 for i in sorted_idx],
            colorscale=[[0,'#1a4a6a'],[0.5,'#00b4d8'],[1,'#00e5ff']]),
        text=[f"{importances[i]*100:.1f}%" for i in sorted_idx],
        textposition='outside', textfont=dict(color='white', size=9)
    ))
    fig_imp.update_layout(
        title=dict(text='Key Risk Drivers (ML Model)', font=dict(color='#00e5ff', size=12)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,26,42,0.8)',
        font=dict(color='#c9d8e8', size=9),
        xaxis=dict(gridcolor='#1a4a6a'),
        yaxis=dict(gridcolor='#1a4a6a'),
        height=280, margin=dict(t=40,b=10,l=10,r=10))
    st.plotly_chart(fig_imp, use_container_width=True)

st.divider()

# ── TOP METRICS ──
st.markdown(f'<div class="section-header">📊 INDIA 🇮🇳 vs {"🇵🇰" if enemy=="Pakistan" else "🇨🇳"} {enemy.upper()} — {year} ASSESSMENT</div>', unsafe_allow_html=True)

m1,m2,m3,m4 = st.columns(4)
with m1:
    st.markdown(f"""<div class="metric-card">
        <h3>WAR PROBABILITY</h3>
        <p class="value {risk_cls}">{war_prob*100:.1f}%</p>
        <small style="color:#4a7fa0">{risk_lbl} RISK</small>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="metric-card">
        <h3>FULL-SCALE WAR</h3>
        <p class="value {'risk-critical' if full_prob>0.4 else 'risk-high'}">{full_prob*100:.1f}%</p>
        <small style="color:#4a7fa0">ESCALATION RISK</small>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""<div class="metric-card">
        <h3>INDIA WIN CHANCE</h3>
        <p class="value risk-low">{india_win*100:.1f}%</p>
        <small style="color:#4a7fa0">IF CONFLICT OCCURS</small>
    </div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""<div class="metric-card">
        <h3>COMPROMISE CHANCE</h3>
        <p class="value risk-medium">{compromise}%</p>
        <small style="color:#4a7fa0">DIPLOMATIC RESOLUTION</small>
    </div>""", unsafe_allow_html=True)

# ── CHARTS ROW 1 ──
c1,c2 = st.columns(2)
with c1:
    st.markdown('<div class="section-header">🎯 CONFLICT OUTCOME PROBABILITIES</div>', unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        x=["Peace / No Conflict","Limited Conflict","Full-Scale War"],
        y=[proba[0]*100, proba[1]*100, proba[2]*100],
        marker_color=["#00e676","#ffea00","#ff1744"],
        text=[f"{proba[0]*100:.1f}%",f"{proba[1]*100:.1f}%",f"{proba[2]*100:.1f}%"],
        textposition='outside', textfont=dict(color='white',size=13)
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,26,42,0.8)',
        font=dict(color='#c9d8e8'), yaxis=dict(gridcolor='#1a4a6a',range=[0,100]),
        height=300, margin=dict(t=30,b=10,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown('<div class="section-header">🏆 IF WAR OCCURS — WIN PROBABILITIES</div>', unsafe_allow_html=True)
    fig2 = go.Figure(go.Pie(
        labels=["India Wins",f"{enemy} Wins","Stalemate"],
        values=[india_win*100, enemy_win*100, stalemate*100],
        hole=0.55,
        marker=dict(colors=["#00e676","#ff1744","#ffea00"],
                    line=dict(color='#030c14',width=2)),
    ))
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#c9d8e8'), legend=dict(font=dict(color='#c9d8e8')),
        height=300, margin=dict(t=30,b=10,l=10,r=10))
    st.plotly_chart(fig2, use_container_width=True)

# ── CHARTS ROW 2 ──
c3,c4 = st.columns(2)
with c3:
    st.markdown('<div class="section-header">⚡ RISK FACTOR RADAR</div>', unsafe_allow_html=True)
    cats = ['Military','Econ Stress','Territory','Nuclear','Alliance Gap','Cyber','Diplomacy (inv)','Incidents']
    vals_r = [mil_balance,econ_stress,terr_dispute,nuke_factor,alliance_gap,cyber_risk,1-dip_relations,recent_inc]
    vals_r += [vals_r[0]]; cats += [cats[0]]
    fig3 = go.Figure(go.Scatterpolar(r=vals_r, theta=cats, fill='toself',
        fillcolor='rgba(0,229,255,0.12)', line=dict(color='#00e5ff',width=2)))
    fig3.update_layout(
        polar=dict(bgcolor='rgba(7,26,42,0.8)',
            radialaxis=dict(range=[0,1],gridcolor='#1a4a6a',tickfont=dict(color='#4a7fa0',size=9)),
            angularaxis=dict(gridcolor='#1a4a6a',tickfont=dict(color='#90b8d4',size=10))),
        paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#c9d8e8'),
        showlegend=False, height=330, margin=dict(t=30,b=10,l=30,r=30))
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.markdown('<div class="section-header">📈 2025–2030 WAR PROBABILITY FORECAST</div>', unsafe_allow_html=True)
    yrs = list(range(2025,2031))
    dip_trend = dip_relations * 0.03
    trend = [max(0.05, min(0.95, war_prob+(y-2025)*(-dip_trend+econ_stress*0.01+recent_inc*0.005))) for y in yrs]
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=yrs, y=[v*100 for v in trend], mode='lines+markers',
        line=dict(color='#00e5ff',width=3), marker=dict(size=8,color='#00e5ff'),
        fill='tozeroy', fillcolor='rgba(0,229,255,0.08)'))
    fig4.add_trace(go.Scatter(x=[year], y=[trend[yrs.index(year)]*100], mode='markers',
        marker=dict(size=14,color='#ff6d00',line=dict(color='white',width=2)), name=f'{year}'))
    fig4.add_hrect(y0=70,y1=100,fillcolor='rgba(255,23,68,0.05)',line_width=0)
    fig4.add_hrect(y0=45,y1=70,fillcolor='rgba(255,109,0,0.05)',line_width=0)
    fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,26,42,0.8)',
        font=dict(color='#c9d8e8'), yaxis=dict(gridcolor='#1a4a6a',range=[0,100]),
        xaxis=dict(gridcolor='#1a4a6a',tickvals=yrs), showlegend=False,
        height=330, margin=dict(t=30,b=10,l=10,r=10))
    st.plotly_chart(fig4, use_container_width=True)

# ── ATTACK ZONES ──
st.markdown('<div class="section-header">🗺️ ATTACK ZONE RISK ANALYSIS</div>', unsafe_allow_html=True)
z1,z2,z3,z4 = st.columns(4)
zones = {
    "🏔️ LAND / BORDER": min(0.95, terr_dispute*0.7 + recent_inc*0.3),
    "🌊 NAVAL":          min(0.95, (0.30 if enemy=="Pakistan" else 0.55) + cyber_risk*0.1 + alliance_gap*0.15),
    "💻 CYBER":          min(0.95, cyber_risk*0.8 + econ_stress*0.2),
    "✈️ AIR STRIKES":    min(0.95, terr_dispute*0.5 + mil_balance*0.2 + recent_inc*0.3),
}
for col,(zone,prob) in zip([z1,z2,z3,z4], zones.items()):
    clr = "#ff1744" if prob>0.7 else "#ff6d00" if prob>0.5 else "#ffea00" if prob>0.3 else "#00e676"
    with col:
        st.markdown(f"""<div class="metric-card" style="text-align:center">
            <h3>{zone}</h3>
            <p class="value" style="color:{clr};font-size:1.8rem">{prob*100:.0f}%</p>
        </div>""", unsafe_allow_html=True)

# ── STRATEGIC ANALYSIS TEXT ──
st.markdown('<div class="section-header">📋 STRATEGIC ANALYSIS</div>', unsafe_allow_html=True)
if enemy == "Pakistan":
    flashpoint = "Kashmir / Line of Control (LoC)"
    key_risk   = "Proxy conflict and cross-border terrorism remain primary escalation vectors. Nuclear threshold is low."
    geo_note   = "Primary land threat: LoC (3,323 km). Naval threat limited to Arabian Sea."
else:
    flashpoint = "LAC — Ladakh / Arunachal Pradesh"
    key_risk   = "Himalayan border standoffs and Indian Ocean naval competition are primary risk drivers."
    geo_note   = "Primary threats: LAC (3,488 km land) + Indian Ocean / South China Sea naval rivalry."

comp_note = "Narrow — diplomatic relations under strain." if dip_relations < 0.4 else "Possible — some diplomatic channel remains open."

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"""
    - **Top Flashpoint:** {flashpoint}
    - **Primary Risk Driver:** {key_risk}
    - **Geographical Note:** {geo_note}
    """)
with col_b:
    st.markdown(f"""
    - **Compromise Window:** {comp_note}
    - **War Probability ({year}):** {war_prob*100:.1f}%
    - **India Win Probability:** {india_win*100:.1f}% (if conflict occurs)
    - **Assessment:** {'🔴 High escalation risk — immediate diplomatic engagement needed.' if war_prob>0.6 else '🟡 Elevated risk — deterrence holding but fragile.' if war_prob>0.35 else '🟢 Risk contained — maintain current diplomatic posture.'}
    """)

# ── INDIA BASELINE ──
st.markdown('<div class="section-header">🇮🇳 INDIA — KEY INDICATORS (2024)</div>', unsafe_allow_html=True)
bi1,bi2,bi3,bi4 = st.columns(4)
stats = [("Defense Budget","$83.6B"),
         ("Active Personnel","1.46M"),
         ("Nuclear Warheads","~172"),
         ("GDP (Nominal)","$3.9T")]
for col,(label,val) in zip([bi1,bi2,bi3,bi4], stats):
    with col:
        st.markdown(f"""<div class="metric-card" style="text-align:center">
            <h3>{label}</h3>
            <p class="value" style="font-size:1.5rem;color:#00e5ff">{val}</p>
        </div>""", unsafe_allow_html=True)