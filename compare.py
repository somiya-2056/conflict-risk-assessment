import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============ PATHS ============
TRAIN_PATH  = r"D:\drdo\ml_dataset_v2.csv"
DUMMY_PATH  = r"D:\drdo\dummy_test_dataset.csv"
OUTPUT_PATH = r"D:\drdo"

# ============ SIMPLE MODEL - sklearn nahi, khud banate hain ============
# Training data load
train_df = pd.read_csv(TRAIN_PATH)

def risk_category(score):
    if score <= 50: return 0
    elif score <= 70: return 1
    else: return 2

train_df['Risk_Category'] = train_df['Conflict_Risk_Score'].apply(risk_category)

FEATURES = ['Military_Ratio', 'Nuclear_Weapons', 'Kashmir_Tension',
            'LAC_Tension', 'Cyber_Attack_Risk', 'Trade_Dependency',
            'Diplomatic_Relations', 'Land_Attack_Risk', 'Naval_Attack_Risk']

# ============ RULE-BASED MODEL (no sklearn needed) ============
# Yeh simple weighted scoring model hai
def predict_risk(row):
    score = 0
    
    # Kashmir tension (India-Pak main factor)
    score += row['Kashmir_Tension'] * 8
    
    # LAC tension (India-China main factor)  
    score += row['LAC_Tension'] * 8
    
    # Cyber risk
    score += row['Cyber_Attack_Risk'] * 5
    
    # Land attack risk
    score += row['Land_Attack_Risk'] * 6
    
    # Naval risk
    score += row['Naval_Attack_Risk'] * 4
    
    # Diplomacy - jitna achha utna kam risk
    score -= row['Diplomatic_Relations'] * 5
    
    # Trade dependency - zyada trade = kam war chance
    score -= row['Trade_Dependency'] * 3
    
    # Military ratio - agar India bahut strong = deterrence
    if row['Military_Ratio'] > 5:
        score -= 10  # deterrence effect
    else:
        score += 15  # India weaker = more risk
    
    # Nuclear weapons of enemy
    score += row['Nuclear_Weapons'] * 0.05
    
    # Normalize to 0-100
    score = max(0, min(100, score))
    
    if score >= 65: return 2, score    # HIGH
    elif score >= 40: return 1, score  # MEDIUM
    else: return 0, score              # LOW

# ============ DUMMY DATA LOAD + PREDICT ============
dummy_df = pd.read_csv(DUMMY_PATH)
label_map = {'HIGH': 2, 'MEDIUM': 1, 'LOW': 0}
risk_names = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}

predictions = []
scores_list = []
for _, row in dummy_df.iterrows():
    pred, sc = predict_risk(row)
    predictions.append(pred)
    scores_list.append(sc)

dummy_df['Model_Predicted'] = [risk_names[p] for p in predictions]
dummy_df['Model_Score']     = scores_list
dummy_df['Match']           = dummy_df['Expected_Risk'] == dummy_df['Model_Predicted']

# Probabilities simulate karo from score
def score_to_proba(sc):
    high   = max(0, (sc - 50) / 50) if sc > 50 else 0
    low    = max(0, (50 - sc) / 50) if sc < 50 else 0
    medium = 1 - high - low
    medium = max(0, medium)
    total  = high + low + medium
    return round(low/total*100,1), round(medium/total*100,1), round(high/total*100,1)

dummy_df['Prob_LOW']    = [score_to_proba(s)[0] for s in scores_list]
dummy_df['Prob_MEDIUM'] = [score_to_proba(s)[1] for s in scores_list]
dummy_df['Prob_HIGH']   = [score_to_proba(s)[2] for s in scores_list]

correct = dummy_df['Match'].sum()
total   = len(dummy_df)
print(f"Dummy Test Score: {correct}/{total} ({correct/total*100:.0f}%)")

# Print results
print("\n" + "="*65)
for _, row in dummy_df.iterrows():
    status = "✅ CORRECT" if row['Match'] else "❌ WRONG"
    print(f"[{row['Scenario_ID']}] {row['Scenario_Name'][:40]}")
    print(f"  Expected: {row['Expected_Risk']:6s} | Model: {row['Model_Predicted']:6s} | Score: {row['Model_Score']:.0f} {status}")
print("="*65)

color_map = {'HIGH': '#e74c3c', 'MEDIUM': '#f39c12', 'LOW': '#27ae60'}

# ============================================================
# GRAPH 1 — Expected vs Predicted
# ============================================================
fig1, ax1 = plt.subplots(figsize=(14, 6))

x     = np.arange(total)
width = 0.35

exp_colors  = [color_map[r] for r in dummy_df['Expected_Risk']]
pred_colors = [color_map[r] for r in dummy_df['Model_Predicted']]

ax1.bar(x - width/2, dummy_df['Expected_Risk'].map(label_map) + 1,
        width, color=exp_colors, alpha=0.9, edgecolor='white', label='Expected')
ax1.bar(x + width/2, dummy_df['Model_Predicted'].map(label_map) + 1,
        width, color=pred_colors, alpha=0.5,
        edgecolor='black', linewidth=0.8, label='Model Predicted')

for i, (_, row) in enumerate(dummy_df.iterrows()):
    mark = '✓' if row['Match'] else '✗'
    clr  = 'green' if row['Match'] else 'red'
    ax1.text(i, 3.6, mark, ha='center', fontsize=13, color=clr, fontweight='bold')

short_names = [s.split(': ')[1][:22] if ': ' in s else s[:22]
               for s in dummy_df['Scenario_Name']]
ax1.set_xticks(x)
ax1.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
ax1.set_yticks([1, 2, 3])
ax1.set_yticklabels(['LOW', 'MEDIUM', 'HIGH'], fontsize=10)
ax1.set_title('Expected Risk vs Model Predicted Risk — All 17 Scenarios',
              fontsize=13, fontweight='bold', pad=15)
ax1.set_ylabel('Risk Level', fontsize=11)
ax1.set_ylim(0, 4.2)

red_p  = mpatches.Patch(color='#e74c3c', label='HIGH')
org_p  = mpatches.Patch(color='#f39c12', label='MEDIUM')
grn_p  = mpatches.Patch(color='#27ae60', label='LOW')
exp_p  = mpatches.Patch(color='gray', alpha=0.9, label='Expected (solid)')
pred_p = mpatches.Patch(color='gray', alpha=0.4, label='Predicted (faded)')
ax1.legend(handles=[red_p, org_p, grn_p, exp_p, pred_p],
           loc='upper right', fontsize=8, ncol=2)
ax1.text(0.01, 0.97, f'Score: {correct}/{total} ({correct/total*100:.0f}%)',
         transform=ax1.transAxes, fontsize=10, va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax1.grid(axis='y', alpha=0.2)

plt.tight_layout()
p1 = OUTPUT_PATH + r"\graph1_expected_vs_predicted.png"
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.show()
print("Graph 1 saved!")

# ============================================================
# GRAPH 2 — Confidence bars
# ============================================================
fig2, axes = plt.subplots(2, 1, figsize=(14, 10))

for pair, ax in zip(['India-Pakistan', 'India-China'], axes):
    sub = dummy_df[dummy_df['Country_Pair'] == pair].reset_index(drop=True)
    x2  = np.arange(len(sub))
    w   = 0.28

    ax.bar(x2 - w, sub['Prob_LOW'],    w, color='#27ae60', label='P(LOW)')
    ax.bar(x2,     sub['Prob_MEDIUM'], w, color='#f39c12', label='P(MEDIUM)')
    ax.bar(x2 + w, sub['Prob_HIGH'],   w, color='#e74c3c', label='P(HIGH)')

    for i, (_, row) in enumerate(sub.iterrows()):
        mark = '✓' if row['Match'] else '✗'
        clr  = 'green' if row['Match'] else 'red'
        ax.text(i, 103,
                f"{mark}\n{row['Expected_Risk'][0]}→{row['Model_Predicted'][0]}",
                ha='center', fontsize=8, color=clr, fontweight='bold')

    short = [s.split(': ')[1][:20] if ': ' in s else s[:20]
             for s in sub['Scenario_Name']]
    ax.set_xticks(x2)
    ax.set_xticklabels(short, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('Probability %', fontsize=10)
    ax.set_ylim(0, 120)
    ax.set_title(f'{pair} — Model Confidence per Scenario',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.axhline(50, color='gray', linestyle=':', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Model Confidence (Probability) — Dummy Test Scenarios',
             fontsize=13, fontweight='bold')
plt.tight_layout()
p2 = OUTPUT_PATH + r"\graph2_confidence.png"
plt.savefig(p2, dpi=150, bbox_inches='tight')
plt.show()
print("Graph 2 saved!")

# ============================================================
# GRAPH 3 — Summary
# ============================================================
fig3, axes3 = plt.subplots(1, 3, figsize=(14, 5))

# Pie
axes3[0].pie([correct, total - correct],
             labels=[f'Correct\n{correct}', f'Wrong\n{total-correct}'],
             colors=['#27ae60', '#e74c3c'],
             autopct='%1.0f%%', startangle=90,
             textprops={'fontsize': 12})
axes3[0].set_title('Overall Accuracy\n(All 17 Scenarios)', fontweight='bold')

# By category
cats   = ['HIGH', 'MEDIUM', 'LOW']
scores_cat = []
for c in cats:
    sub = dummy_df[dummy_df['Expected_Risk'] == c]
    scores_cat.append(sub['Match'].mean() * 100 if len(sub) > 0 else 0)

bars = axes3[1].bar(cats, scores_cat,
                    color=['#e74c3c', '#f39c12', '#27ae60'],
                    edgecolor='white', width=0.5)
for bar, sc in zip(bars, scores_cat):
    axes3[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                  f'{sc:.0f}%', ha='center', fontsize=11, fontweight='bold')
axes3[1].set_ylim(0, 115)
axes3[1].set_ylabel('Accuracy %')
axes3[1].set_title('Accuracy by\nRisk Category', fontweight='bold')
axes3[1].axhline(50, color='gray', linestyle='--', alpha=0.5)
axes3[1].grid(axis='y', alpha=0.3)

# By country pair
pairs       = ['India-Pakistan', 'India-China']
pair_scores = [dummy_df[dummy_df['Country_Pair'] == p]['Match'].mean() * 100
               for p in pairs]
bars2 = axes3[2].bar(['India\nPakistan', 'India\nChina'], pair_scores,
                     color=['#3498db', '#9b59b6'], edgecolor='white', width=0.5)
for bar, sc in zip(bars2, pair_scores):
    axes3[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                  f'{sc:.0f}%', ha='center', fontsize=11, fontweight='bold')
axes3[2].set_ylim(0, 115)
axes3[2].set_ylabel('Accuracy %')
axes3[2].set_title('Accuracy by\nCountry Pair', fontweight='bold')
axes3[2].axhline(50, color='gray', linestyle='--', alpha=0.5)
axes3[2].grid(axis='y', alpha=0.3)

plt.suptitle('Dummy Test — Summary Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
p3 = OUTPUT_PATH + r"\graph3_summary.png"
plt.savefig(p3, dpi=150, bbox_inches='tight')
plt.show()
print("Graph 3 saved!")

print(f"\n✅ Teeno graphs ban gaye aur D:\\drdo mein save ho gaye!")