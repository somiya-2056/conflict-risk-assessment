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
    html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; background-color: #030c14; color: #c9d8e8; }
    .stApp { background-color: #030c14; }
    .main-title { font-family: 'Orbitron', monospace; font-size: 1.8rem; color: #00e5ff; text-align: center; letter-spacing: 3px; text-shadow: 0 0 20px #00e5ff88; padding: 1rem 0 0.2rem 0; }
    .sub-title { font-family: 'Share Tech Mono', monospace; font-size: 0.8rem; color: #4a7fa0; text-align: center; letter-spacing: 4px; margin-bottom: 1rem; }
    .metric-card { background: linear-gradient(135deg, #071a2a 0%, #0a2540 100%); border: 1px solid #1a4a6a; border-radius: 10px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; box-shadow: 0 4px 20px #00000055; }
    .metric-card h3 { font-family: 'Share Tech Mono', monospace; font-size: 0.72rem; color: #4a9aba; letter-spacing: 3px; margin: 0 0 0.5rem 0; text-transform: uppercase; }
    .metric-card .value { font-family: 'Orbitron', monospace; font-size: 2rem; font-weight: 700; margin: 0; }
    .risk-low      { color: #00e676; text-shadow: 0 0 12px #00e67688; }
    .risk-medium   { color: #ffea00; text-shadow: 0 0 12px #ffea0088; }
    .risk-high     { color: #ff6d00; text-shadow: 0 0 12px #ff6d0088; }
    .risk-critical { color: #ff1744; text-shadow: 0 0 16px #ff174488; }
    .section-header { font-family: 'Share Tech Mono', monospace; font-size: 0.78rem; color: #00b4d8; letter-spacing: 4px; border-bottom: 1px solid #1a4a6a; padding-bottom: 0.4rem; margin: 1.5rem 0 1rem 0; text-transform: uppercase; }
    hr { border-color: #1a4a6a; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── FEATURES ORDER — must match training AND sliders below ──
FEATURES = [
    'Military_Ratio', 'Nuclear_Weapons', 'Historical_Wars',
    'Kashmir_Tension', 'LAC_Tension', 'Cyber_Attack_Risk',
    'Trade_Dependency', 'Diplomatic_Relations',
    'Land_Attack_Risk', 'Naval_Attack_Risk'
]

@st.cache_resource
def train_models():
    df = pd.read_csv(r"D:\drdo\historical_dataset_1947_2024.csv")
    from sklearn.utils import resample

    def risk_cat(score):
        if score <= 50: return 0
        elif score <= 70: return 1
        else: return 2

    df['Risk_Category'] = df['Conflict_Risk_Score'].apply(risk_cat)

    df_high = df[df['Risk_Category']==2]
    df_med  = df[df['Risk_Category']==1]
    df_low  = df[df['Risk_Category']==0]
    target  = max(len(df_high), len(df_med), len(df_low))

    df_bal = pd.concat([
        resample(df_high, replace=True, n_samples=target, random_state=42),
        resample(df_med,  replace=True, n_samples=target, random_state=42),
        resample(df_low,  replace=True, n_samples=target, random_state=42),
    ]).reset_index(drop=True)

    X = df_bal[FEATURES]
    y = df_bal['Risk_Category']

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=3,
                                 class_weight='balanced', random_state=42)
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                     max_depth=4, random_state=42)
    rf.fit(X_sc, y)
    gb.fit(X_sc, y)

    rf_cv = cross_val_score(rf, X_sc, y, cv=5).mean()
    gb_cv = cross_val_score(gb, X_sc, y, cv=5).mean()
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

# features = [mil_ratio, nuclear, hist_wars, kashmir, lac, cyber, trade, dip, land, naval]
def india_win_prob(features, enemy):
    mil, nuke, wars, kashmir, lac, cyber, trade, dip, land, naval = features
    base = 0.55
    if enemy == "Pakistan":
        base += 0.10 + (mil-1)*0.03 - (kashmir/10)*0.05
    else:
        base -= 0.12 + (mil-0.3)*0.05 - (lac/10)*0.05
    base -= (nuke/400)*0.08
    base += (dip/10)*0.10
    return max(0.10, min(0.88, base))

# ── HEADER ──
st.markdown('<div class="main-title">🛡️ AI-BASED CONFLICT RISK ASSESSMENT</div>', unsafe_allow_html=True)
st.divider()

# ── INPUTS — directly mapped to FEATURES order ──
st.markdown('<div class="section-header">⚙️ SCENARIO INPUTS — ADJUST INDICATORS</div>', unsafe_allow_html=True)
inp1, inp2 = st.columns(2)
with inp1: enemy = st.selectbox("India vs →", ["Pakistan", "China"])
with inp2: year  = st.selectbox("Projection Year", [2026,2027,2028,2029,2030])

st.markdown("<br>", unsafe_allow_html=True)
col_m, col_g, col_e, col_d = st.columns(4)

with col_m:
    st.markdown("**🪖 Military**")
    mil_ratio = st.slider("Military Ratio (India/Enemy)", 0.1, 10.0,
                           5.0 if enemy=="Pakistan" else 0.3, 0.1)
    nuclear   = st.slider("Enemy Nuclear Weapons", 50, 400,
                           170 if enemy=="Pakistan" else 350, 10)
    hist_wars = st.slider("Historical Wars Count", 0, 5,
                           4 if enemy=="Pakistan" else 1, 1)

with col_g:
    st.markdown("**🌐 Geopolitical**")
    kashmir_tension = st.slider("Kashmir Tension (LoC)", 0.0, 10.0,
                                 7.0 if enemy=="Pakistan" else 0.0, 0.5)
    lac_tension     = st.slider("LAC Tension (Border)",  0.0, 10.0,
                                 7.0 if enemy=="China" else 0.0, 0.5)
    

with col_e:
    st.markdown("**💰 Economic**")
    trade_dep     = st.slider("Trade Dependency",     0.0, 10.0,
                               1.0 if enemy=="Pakistan" else 6.0, 0.5)
    dip_relations = st.slider("Diplomatic Relations", 0.0, 10.0, 3.0, 0.5)

with col_d:
    st.markdown("**🗺️ Attack Risk**")
    land_risk  = st.slider("Land Attack Risk",  0.0, 10.0, 7.0, 0.5)
    naval_risk = st.slider("Naval Attack Risk", 0.0, 10.0,
                            3.0 if enemy=="Pakistan" else 6.0, 0.5)
    cyber_risk      = st.slider("Cyber Attack Risk",      0.0, 10.0, 5.0, 0.5)
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("🔍 RUN PREDICTION", use_container_width=True)

st.divider()

# 1:1 mapping with FEATURES — no scrambling!
features = [mil_ratio, nuclear, hist_wars, kashmir_tension, lac_tension,
            cyber_risk, trade_dep, dip_relations, land_risk, naval_risk]

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

# Higher diplomacy + trade + lower tension/incidents = higher compromise chance
compromise = round((dip_relations*4 + trade_dep*3 +
                     (10-kashmir_tension)*1.5 + (10-lac_tension)*1.5) / 10 * 10, 1)
compromise = min(max(compromise, 5), 80)

# ── INDIA HISTORICAL GRAPHS ──
st.markdown('<div class="section-header">📊 INDIA — HISTORICAL STRATEGIC DATA (1947–2025)</div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    sipri_df = pd.read_csv(r"D:\drdo\sipri_full.csv")
    sipri_df = sipri_df.replace('...', None)
    hist_df = pd.read_csv(r"D:\drdo\historical_dataset_1947_2024.csv")
    return sipri_df, hist_df

sipri_df, hist_df = load_data()

mil_years = sipri_df['Year'].tolist()
mil_data = {
    'India':    pd.to_numeric(sipri_df['India'],    errors='coerce').tolist(),
    'Pakistan': pd.to_numeric(sipri_df['Pakistan'], errors='coerce').tolist(),
    'China':    pd.to_numeric(sipri_df['China'],    errors='coerce').tolist(),
}

gdp_df = pd.read_csv(r"D:\drdo\gdp_data.csv")
gdp_years = gdp_df['Year'].astype(int).tolist()
gdp_data = {
    'India':    gdp_df['India'].tolist(),
    'Pakistan': gdp_df['Pakistan'].tolist(),
    'China':    gdp_df['China'].tolist(),
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
        yaxis=dict(gridcolor='#1a4a6a'), xaxis=dict(gridcolor='#1a4a6a'),
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
        yaxis=dict(gridcolor='#1a4a6a'), xaxis=dict(gridcolor='#1a4a6a'),
        legend=dict(font=dict(color='#c9d8e8', size=9), bgcolor='rgba(0,0,0,0)'),
        height=280, margin=dict(t=40,b=10,l=10,r=10))
    st.plotly_chart(fig_gdp, use_container_width=True)

with hg3:
    feat_names = ['Mil Ratio','Nuclear','Hist Wars','Kashmir','LAC',
                   'Cyber','Trade Dep','Diplomacy','Land Risk','Naval Risk']
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)
    fig_imp = go.Figure(go.Bar(
        x=[importances[i]*100 for i in sorted_idx],
        y=[feat_names[i] for i in sorted_idx],
        orientation='h',
        marker=dict(color=[importances[i]*100 for i in sorted_idx],
                    colorscale=[[0,'#1a4a6a'],[0.5,'#00b4d8'],[1,'#00e5ff']]),
        text=[f"{importances[i]*100:.1f}%" for i in sorted_idx],
        textposition='outside', textfont=dict(color='white', size=9)
    ))
    fig_imp.update_layout(
        title=dict(text=f'Key Risk Drivers · CV: {rf_acc*100:.1f}%', font=dict(color='#00e5ff', size=12)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(7,26,42,0.8)',
        font=dict(color='#c9d8e8', size=9),
        xaxis=dict(gridcolor='#1a4a6a'), yaxis=dict(gridcolor='#1a4a6a'),
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
        marker=dict(colors=["#00e676","#ff1744","#ffea00"], line=dict(color='#030c14',width=2)),
    ))
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#c9d8e8'), legend=dict(font=dict(color='#c9d8e8')),
        height=300, margin=dict(t=30,b=10,l=10,r=10))
    st.plotly_chart(fig2, use_container_width=True)

# ── CHARTS ROW 2 ──
c3,c4 = st.columns(2)
with c3:
    st.markdown('<div class="section-header">⚡ RISK FACTOR RADAR</div>', unsafe_allow_html=True)
    cats  = ['Mil Ratio','Nuclear','Hist Wars','Kashmir/LAC','Cyber','Trade','Diplomacy (inv)','Attack Risk']
    main_tension = kashmir_tension if enemy=="Pakistan" else lac_tension
    attack_avg   = (land_risk + naval_risk) / 2
    vals_r = [mil_ratio/10, nuclear/400, hist_wars/5, main_tension/10,
              cyber_risk/10, trade_dep/10, 1-(dip_relations/10), attack_avg/10]
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
    # Higher diplomacy -> declining trend. Higher incidents/tension -> rising trend.
    dip_trend = (dip_relations/10) * 0.03
    tension_trend = (main_tension/10) * 0.01
    trend = [max(0.05, min(0.95, war_prob + (y-2025)*(tension_trend - dip_trend))) for y in yrs]
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
    "🏔️ LAND / BORDER": min(0.95, (land_risk/10)*0.7 + (main_tension/10)*0.3),
    "🌊 NAVAL":          min(0.95, (naval_risk/10)*0.7 + (0.30 if enemy=="Pakistan" else 0.55)*0.3),
    "💻 CYBER":          min(0.95, (cyber_risk/10)*0.8 + (1-trade_dep/10)*0.2),
    "✈️ AIR STRIKES":    min(0.95, (land_risk/10)*0.5 + (mil_ratio/10)*0.2 + (main_tension/10)*0.3),
}
for col,(zone,prob) in zip([z1,z2,z3,z4], zones.items()):
    clr = "#ff1744" if prob>0.7 else "#ff6d00" if prob>0.5 else "#ffea00" if prob>0.3 else "#00e676"
    with col:
        st.markdown(f"""<div class="metric-card" style="text-align:center">
            <h3>{zone}</h3>
            <p class="value" style="color:{clr};font-size:1.8rem">{prob*100:.0f}%</p>
        </div>""", unsafe_allow_html=True)

# ── STRATEGIC ANALYSIS ──
st.markdown('<div class="section-header">📋 STRATEGIC ANALYSIS</div>', unsafe_allow_html=True)
if enemy == "Pakistan":
    flashpoint = "Kashmir / Line of Control (LoC)"
    key_risk   = "Proxy conflict and cross-border terrorism remain primary escalation vectors. Nuclear threshold is low."
    geo_note   = "Primary land threat: LoC (3,323 km). Naval threat limited to Arabian Sea."
else:
    flashpoint = "LAC — Ladakh / Arunachal Pradesh"
    key_risk   = "Himalayan border standoffs and Indian Ocean naval competition are primary risk drivers."
    geo_note   = "Primary threats: LAC (3,488 km land) + Indian Ocean / South China Sea naval rivalry."

comp_note = "Narrow — diplomatic relations under strain." if dip_relations < 4 else "Possible — some diplomatic channel remains open."

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