import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  DRDO CONFLICT RISK MODEL - TRAINING v3")
print("  Dataset: 1947-2024 Historical Data")
print("=" * 60)

# ============ LOAD DATA ============
df = pd.read_csv(r"D:\drdo\historical_dataset_1947_2024.csv")
print(f"\n✅ Dataset loaded: {len(df)} rows, {df.shape[1]} columns")
print(f"   Year range: {df['Year'].min()} - {df['Year'].max()}")

# ============ RISK LABELS ============
def risk_category(score):
    if score <= 50:  return 0  # LOW
    elif score <= 70: return 1  # MEDIUM
    else:             return 2  # HIGH

df['Risk_Category'] = df['Conflict_Risk_Score'].apply(risk_category)
label_names = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}

print(f"\n📊 Class distribution (before balancing):")
for k, v in label_names.items():
    count = (df['Risk_Category'] == k).sum()
    print(f"   {v}: {count} rows")

# ============ BALANCE CLASSES ============
df_high   = df[df['Risk_Category'] == 2]
df_medium = df[df['Risk_Category'] == 1]
df_low    = df[df['Risk_Category'] == 0]

target = max(len(df_high), len(df_medium), len(df_low))

df_high_bal   = resample(df_high,   replace=True, n_samples=target, random_state=42)
df_medium_bal = resample(df_medium, replace=True, n_samples=target, random_state=42)
df_low_bal    = resample(df_low,    replace=True, n_samples=target, random_state=42)

df_balanced = pd.concat([df_high_bal, df_medium_bal, df_low_bal]).reset_index(drop=True)

print(f"\n✅ After balancing: {len(df_balanced)} rows (each class = {target})")

# ============ FEATURES ============
FEATURES = [
    'Military_Ratio', 'Nuclear_Weapons', 'Historical_Wars',
    'Kashmir_Tension', 'LAC_Tension', 'Cyber_Attack_Risk',
    'Trade_Dependency', 'Diplomatic_Relations',
    'Land_Attack_Risk', 'Naval_Attack_Risk'
]

X = df_balanced[FEATURES]
y = df_balanced['Risk_Category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📐 Train size: {len(X_train)}, Test size: {len(X_test)}")

# ============ TRAIN MODELS ============
print("\n🔧 Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=3,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train, y_train)

print("🔧 Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
gb.fit(X_train, y_train)

# ============ ACCURACY ============
rf_train = accuracy_score(y_train, rf.predict(X_train))
rf_test  = accuracy_score(y_test,  rf.predict(X_test))
gb_test  = accuracy_score(y_test,  gb.predict(X_test))

print(f"\n📈 RESULTS:")
print(f"   Random Forest  - Train: {rf_train*100:.1f}%  |  Test: {rf_test*100:.1f}%")
print(f"   Gradient Boost - Test:  {gb_test*100:.1f}%")
print(f"   Overfit gap:    {(rf_train - rf_test)*100:.1f}% (below 15% = acceptable)")

# ============ CROSS VALIDATION ============
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
print(f"\n🔁 Cross Validation (5-fold):")
print(f"   Scores: {[f'{s*100:.1f}%' for s in cv_scores]}")
print(f"   Average: {cv_scores.mean()*100:.1f}%  ±  {cv_scores.std()*100:.1f}%")

# ============ CLASSIFICATION REPORT ============
y_pred = rf.predict(X_test)
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['LOW', 'MEDIUM', 'HIGH']))

# ============ FEATURE IMPORTANCE ============
importance = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"\n🎯 Feature Importance:")
for feat, imp in importance.items():
    bar = '█' * int(imp * 50)
    print(f"   {feat:<25} {bar}  {imp*100:.1f}%")

# ============ SAVE MODEL ============
with open(r"D:\drdo\conflict_model.pkl", 'wb') as f:
    pickle.dump({'rf': rf, 'gb': gb, 'features': FEATURES}, f)
print(f"\n💾 Model saved: D:\\drdo\\conflict_model.pkl")

# ============ GRAPHS ============
# Graph 1 - Feature Importance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ['#d32f2f' if i == 0 else '#1976d2' if i == 1 else '#388e3c' for i in range(len(importance))]
axes[0].barh(importance.index[::-1], importance.values[::-1], color=colors[::-1])
axes[0].set_title('Feature Importance — Which Factor Matters Most?', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Importance Score')
for i, (feat, val) in enumerate(zip(importance.index[::-1], importance.values[::-1])):
    axes[0].text(val + 0.002, i, f'{val*100:.1f}%', va='center', fontsize=9)

# Graph 2 - Risk distribution over time (original data)
df_plot = df.copy()
year_avg_ip = df_plot[df_plot['Country_Pair']=='India-Pakistan'].groupby('Year')['Conflict_Risk_Score'].mean()
year_avg_ic = df_plot[df_plot['Country_Pair']=='India-China'].groupby('Year')['Conflict_Risk_Score'].mean()

axes[1].plot(year_avg_ip.index, year_avg_ip.values, color='red', linewidth=2, label='India-Pakistan')
axes[1].plot(year_avg_ic.index, year_avg_ic.values, color='blue', linewidth=2, label='India-China')
axes[1].axhline(y=70, color='orange', linestyle='--', alpha=0.7, label='HIGH threshold (70)')
axes[1].axhline(y=50, color='green',  linestyle='--', alpha=0.7, label='LOW threshold (50)')
axes[1].fill_between(year_avg_ip.index, 70, 100, alpha=0.08, color='red')

# Mark major wars
war_years = {'1947\nPartition': (1947, 95), '1965\nWar': (1965, 96),
             '1971\nWar': (1971, 98), '1999\nKargil': (1999, 96),
             '1962\nWar': (1962, 97)}
for label, (yr, score) in war_years.items():
    axes[1].annotate(label, xy=(yr, score), xytext=(yr, score+2),
                    fontsize=7, ha='center', color='darkred',
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=0.8))

axes[1].set_title('Conflict Risk Score — 1947 to 2024', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Risk Score (0-100)')
axes[1].legend(loc='lower right', fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(15, 110)

plt.tight_layout()
plt.savefig(r"D:\drdo\model_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"📊 Graph saved: D:\\drdo\\model_analysis.png")

# ============ DUMMY TEST ============
print(f"\n{'='*60}")
print("  DUMMY TESTING — HISTORICAL SCENARIOS")
print(f"{'='*60}")

scenarios = [
    # name, values, expected
    ("1947 Partition War (India-Pak)",
     [3.0, 0, 1, 10, 0, 0, 1.0, 1, 10, 1], "HIGH"),
    ("1962 India-China War",
     [0.9, 0, 0,  0,10, 0, 0.0, 1, 10, 2], "HIGH"),
    ("1999 Kargil War",
     [9.5,165, 4, 10, 0, 1, 1.0, 1, 10, 3], "HIGH"),
    ("2021 LOC Ceasefire (India-Pak)",
     [9.0,165, 4,  6, 0, 8, 1.5, 3,  5, 2], "MEDIUM"),
    ("2018 Wuhan Summit (India-China)",
     [0.28,350,1, 0, 4, 6, 7.0, 6,  4, 5], "LOW"),
    ("2020 Galwan Clash (India-China)",
     [0.27,350,1, 0, 8, 8, 5.0, 2,  9, 7], "HIGH"),
    ("2003 LOC Ceasefire (India-Pak)",
     [8.5,140, 4,  7, 0, 2, 1.0, 3,  6, 2], "MEDIUM"),
    ("2024 Post-Disengagement (India-China)",
     [0.27,350,1, 0, 5, 8, 8.0, 5,  5, 7], "LOW"),
]

correct = 0
for name, values, expected in scenarios:
    inp = pd.DataFrame([values], columns=FEATURES)
    pred = rf.predict(inp)[0]
    prob = rf.predict_proba(inp)[0]
    result = label_names[pred]
    match = "✅" if result == expected else "⚠️"
    if result == expected: correct += 1
    print(f"\n  {match} {name}")
    print(f"     Expected: {expected:6s}  |  Model: {result:6s}")
    print(f"     Confidence — LOW:{prob[0]*100:.0f}% MED:{prob[1]*100:.0f}% HIGH:{prob[2]*100:.0f}%")

score_pct = correct / len(scenarios) * 100
print(f"\n{'='*60}")
print(f"  DUMMY TEST SCORE: {correct}/{len(scenarios)} = {score_pct:.0f}%")
if score_pct >= 75:
    print("  ✅ Model performance: GOOD")
elif score_pct >= 60:
    print("  ⚠️  Model performance: ACCEPTABLE")
else:
    print("  ❌ Model needs improvement")
print(f"{'='*60}")
print("\n✅ Training complete!")