import pandas as pd
import numpy as np

# ============================================================
# HISTORICAL DATASET 1947-2024
# 156 rows total (78 India-Pak + 78 India-China)
# Features: 16 columns + Real_Event + Risk_Label
# ============================================================

rows = []

# Military ratio (India/Enemy) - researched values
# India-Pak ratio goes from ~3x (1947) to ~9x (2024)
# India-China ratio goes from ~1.2x (1947) to ~0.25x (2024) [China grew faster]

ip_mil_ratio = {
    range(1947,1956): 3.0, range(1956,1962): 4.0, range(1962,1970): 5.0,
    range(1970,1980): 5.5, range(1980,1990): 6.5, range(1990,2000): 7.5,
    range(2000,2010): 8.5, range(2010,2020): 9.0, range(2020,2025): 9.5
}

ic_mil_ratio = {
    range(1947,1955): 1.2, range(1955,1962): 0.9, range(1962,1975): 0.5,
    range(1975,1985): 0.45, range(1985,1995): 0.40, range(1995,2005): 0.35,
    range(2005,2015): 0.30, range(2015,2025): 0.27
}

def get_ratio(year, ratio_dict):
    for r, val in ratio_dict.items():
        if year in r:
            return val
    return 0.3

# ---- INDIA - PAKISTAN ----
# Format: kashmir, diplomacy, land_risk, naval_risk, cyber, compromise, risk_score, event
ip_events = {
    1947: (10,1,10,1,0,2, 95,"Partition + First Kashmir War"),
    1948: (9, 1, 9,1,0,3, 90,"Ceasefire after First Kashmir War"),
    1949: (8, 2, 7,1,0,4, 80,"Karachi Agreement on ceasefire line"),
    1950: (6, 3, 5,1,0,5, 65,"Relative calm, UN Kashmir debates"),
    1951: (6, 3, 5,1,0,5, 63,"Relative calm"),
    1952: (5, 4, 4,1,0,6, 58,"UN resolutions on Kashmir"),
    1953: (5, 4, 4,1,0,6, 57,"Nehru-Bogra talks"),
    1954: (5, 4, 4,1,0,6, 56,"Pakistan joins SEATO"),
    1955: (6, 3, 5,1,0,5, 62,"Pakistan joins Baghdad Pact"),
    1956: (6, 3, 5,2,0,5, 63,"Tension rising"),
    1957: (6, 3, 5,2,0,5, 62,"UN Security Council debates"),
    1958: (6, 3, 5,2,0,5, 61,"Ayub Khan military coup"),
    1959: (6, 3, 5,2,0,5, 62,"Tension"),
    1960: (5, 4, 4,2,0,6, 58,"Indus Waters Treaty signed - de-escalation"),
    1961: (5, 4, 4,2,0,6, 57,"Relative stability"),
    1962: (6, 3, 5,2,0,5, 63,"India-China war - Pakistan watching"),
    1963: (7, 2, 7,2,0,4, 72,"Pak-China boundary agreement - threat to India"),
    1964: (7, 2, 7,2,0,3, 74,"Operation Gibraltar planning begins"),
    1965: (10,1,10,3,0,2, 96,"1965 WAR - Operation Grand Slam"),
    1966: (8, 2, 7,2,0,3, 82,"Tashkent Declaration post-war"),
    1967: (7, 3, 6,2,0,4, 72,"Relative calm"),
    1968: (7, 3, 6,2,0,4, 71,"Tension continues"),
    1969: (7, 3, 6,2,0,4, 70,"Yahya Khan takes power"),
    1970: (8, 2, 7,3,0,3, 78,"Bangladesh crisis building"),
    1971: (10,1,10,8,0,1, 98,"1971 WAR - Bangladesh Liberation"),
    1972: (8, 2, 8,4,0,3, 82,"Simla Agreement signed"),
    1973: (7, 3, 6,3,0,4, 70,"Post-war normalization"),
    1974: (7, 3, 6,3,0,4, 71,"India Pokhran-I nuclear test"),
    1975: (6, 4, 5,2,0,5, 64,"Relative stability"),
    1976: (6, 4, 5,2,0,5, 63,"Diplomatic relations restored"),
    1977: (6, 4, 5,2,0,5, 62,"Zia ul-Haq military coup"),
    1978: (6, 3, 5,2,0,4, 65,"Tension"),
    1979: (7, 3, 6,2,0,4, 70,"Soviet invasion Afghanistan - Pak frontline"),
    1980: (7, 3, 6,2,0,4, 71,"Cold War proxy dynamics"),
    1981: (7, 3, 6,2,0,4, 70,"Tension"),
    1982: (7, 3, 6,2,0,4, 71,"Tension"),
    1983: (7, 3, 6,2,0,4, 70,"Punjab insurgency starts"),
    1984: (8, 2, 7,2,0,3, 78,"Operation Blue Star + Siachen conflict"),
    1985: (7, 3, 6,2,0,4, 72,"SAARC formed"),
    1986: (8, 2, 7,2,0,3, 78,"Operation Brasstacks crisis"),
    1987: (9, 1, 8,2,0,2, 85,"Brasstacks near-war standoff"),
    1988: (7, 3, 6,2,0,4, 70,"Benazir Bhutto - Rajiv Gandhi talks"),
    1989: (8, 2, 7,2,0,3, 78,"Kashmir insurgency begins"),
    1990: (9, 1, 8,2,0,2, 86,"Near-nuclear war scare"),
    1991: (8, 2, 7,2,0,3, 79,"Kashmir insurgency peak"),
    1992: (8, 2, 7,2,0,3, 80,"Cross-border terrorism high"),
    1993: (8, 2, 7,2,0,3, 79,"Hazratbal siege"),
    1994: (8, 2, 7,2,0,3, 79,"Continued insurgency"),
    1995: (8, 2, 7,2,0,3, 78,"Al-Faran kidnappings"),
    1996: (7, 3, 6,2,0,4, 72,"Slight improvement"),
    1997: (7, 3, 6,2,1,4, 70,"Composite dialogue starts"),
    1998: (9, 1, 8,3,1,2, 87,"Nuclear tests both sides - Pokhran II + Chagai"),
    1999: (10,1,10,3,1,2, 96,"KARGIL WAR"),
    2000: (8, 2, 7,2,2,3, 80,"Post-Kargil tension"),
    2001: (9, 1, 8,2,2,2, 86,"Parliament attack - Operation Parakram begins"),
    2002: (9, 1, 9,2,2,2, 88,"Operation Parakram - million troops mobilized"),
    2003: (7, 3, 6,2,2,4, 70,"LOC Ceasefire November 2003"),
    2004: (6, 5, 5,2,2,5, 62,"Composite dialogue resumes"),
    2005: (6, 4, 5,2,3,5, 63,"Bus diplomacy Srinagar-Muzaffarabad"),
    2006: (7, 3, 6,2,3,4, 71,"Mumbai train blasts - 7/11"),
    2007: (7, 3, 6,2,3,4, 70,"Samjhauta Express blast"),
    2008: (9, 1, 8,2,4,2, 87,"26/11 Mumbai attacks"),
    2009: (8, 2, 7,2,4,3, 79,"Post-26/11 tension"),
    2010: (7, 3, 6,2,4,4, 71,"Talks resume"),
    2011: (7, 3, 6,2,5,4, 70,"MFN status discussion"),
    2012: (7, 3, 6,2,5,4, 69,"Trade talks"),
    2013: (8, 2, 7,2,5,3, 77,"LOC violations increase significantly"),
    2014: (7, 3, 6,2,6,4, 70,"Modi-Sharif Oath ceremony diplomacy"),
    2015: (7, 3, 6,2,6,4, 69,"Ufa meeting"),
    2016: (9, 1, 8,2,7,2, 87,"URI attack + Surgical Strike"),
    2017: (8, 2, 7,2,7,3, 79,"Tension continues post-Uri"),
    2018: (7, 3, 6,2,7,4, 70,"Kartarpur corridor talks"),
    2019: (9, 1, 8,2,8,2, 88,"Pulwama + Balakot airstrike + Article 370"),
    2020: (8, 2, 7,2,8,3, 80,"COVID - reduced cross-border activity"),
    2021: (6, 3, 5,2,8,5, 65,"LOC Ceasefire February 2021"),
    2022: (7, 3, 6,2,8,4, 71,"Tension resumes"),
    2023: (7, 3, 6,2,8,4, 72,"Continued tension"),
    2024: (7, 3, 6,2,9,4, 71,"India elections - Pakistan political crisis"),
}

# ---- INDIA - CHINA ----
# Format: lac_tension, diplomacy, land_risk, naval_risk, cyber, compromise, risk_score, event
ic_events = {
    1947: (1, 5,2,1,0,7, 28,"Both newly independent - early neutrality"),
    1948: (1, 5,2,1,0,7, 28,"Hindi-Chini bhai bhai era begins"),
    1949: (1, 6,2,1,0,7, 25,"PRC founded - India first to recognize"),
    1950: (2, 5,3,1,0,6, 32,"China occupies Tibet - India concerned"),
    1951: (2, 5,3,1,0,6, 33,"Tibet formally annexed"),
    1952: (2, 5,3,1,0,6, 32,"Panchsheel discussions begin"),
    1953: (2, 6,3,1,0,7, 28,"Panchsheel - 5 principles of coexistence"),
    1954: (2, 7,2,1,0,8, 25,"Panchsheel Agreement signed - peak friendship"),
    1955: (2, 7,2,1,0,8, 24,"Bandung Conference - Nehru-Zhou Enlai"),
    1956: (3, 6,3,1,0,7, 32,"Border disputes emerging"),
    1957: (4, 5,4,1,0,6, 40,"China secretly builds road in Aksai Chin"),
    1958: (5, 4,5,1,0,5, 48,"India discovers Aksai Chin road - shock"),
    1959: (7, 3,7,1,0,4, 68,"Tibet uprising - Dalai Lama flees to India"),
    1960: (7, 3,7,1,0,3, 70,"Border talks fail completely"),
    1961: (8, 2,8,1,0,2, 78,"Forward Policy - India sends patrols"),
    1962: (10,1,10,2,0,1, 97,"1962 WAR - India badly defeated"),
    1963: (8, 2,7,2,0,2, 80,"Post-war - diplomatic relations suspended"),
    1964: (7, 2,6,2,0,2, 72,"China nuclear test - India alarmed"),
    1965: (7, 2,6,2,0,2, 70,"China threatens India during 65 Pak war"),
    1966: (6, 2,5,2,0,3, 65,"Cultural revolution - China inward focus"),
    1967: (7, 2,7,2,0,3, 68,"Nathu La and Cho La border clashes"),
    1968: (6, 2,5,2,0,3, 63,"Cultural revolution continues"),
    1969: (5, 2,5,2,0,3, 58,"Relations strained"),
    1970: (5, 3,4,2,0,4, 52,"Slight improvement"),
    1971: (6, 2,5,3,0,3, 62,"China supports Pakistan in 1971 war"),
    1972: (5, 3,4,2,0,4, 52,"Nixon visits China - new global dynamics"),
    1973: (5, 3,4,2,0,4, 50,"Gradual normalization"),
    1974: (5, 3,4,2,0,4, 52,"India nuclear test - China strongly protests"),
    1975: (4, 4,4,2,0,5, 46,"Ambassador restored after 15 years"),
    1976: (4, 4,4,2,0,5, 45,"Normalization continues slowly"),
    1977: (4, 4,4,2,0,5, 45,"Talks"),
    1978: (4, 4,4,2,0,5, 46,"Vajpayee visits China - first senior visit"),
    1979: (5, 3,5,2,0,4, 52,"Sino-Vietnamese war - India neutral"),
    1980: (5, 3,5,2,0,4, 50,"Border talks formally begin"),
    1981: (4, 4,4,2,0,5, 47,"Border talks round 1"),
    1982: (4, 4,4,2,0,5, 46,"Border talks round 2"),
    1983: (4, 4,4,2,0,5, 46,"Border talks continue"),
    1984: (5, 3,5,2,0,4, 52,"Sumdorong Chu incident"),
    1985: (5, 3,5,2,0,4, 51,"Continued tension"),
    1986: (6, 3,6,2,0,4, 62,"Sumdorong Chu standoff escalates"),
    1987: (6, 3,6,2,0,4, 63,"Wangdung standoff - near conflict"),
    1988: (4, 5,4,2,0,6, 45,"Rajiv Gandhi visits China - landmark reset"),
    1989: (4, 5,4,2,0,6, 44,"Tiananmen - India stays neutral"),
    1990: (4, 5,4,2,0,6, 44,"Stable bilateral ties"),
    1991: (4, 5,4,2,0,6, 43,"India economic reforms - China trade begins"),
    1992: (4, 5,4,2,0,6, 43,"Narasimha Rao visits China"),
    1993: (3, 6,3,3,0,7, 38,"Border Peace and Tranquility Agreement"),
    1994: (3, 6,3,3,0,7, 37,"CBMs implementation"),
    1995: (3, 6,3,3,0,7, 36,"Stable"),
    1996: (3, 6,3,3,0,7, 36,"Jiang Zemin visits India"),
    1997: (3, 6,3,3,1,7, 36,"Stable - growing trade"),
    1998: (6, 3,6,3,1,4, 64,"India nuclear test - China strongly opposes"),
    1999: (5, 4,5,3,1,5, 53,"Kargil - China stays carefully neutral"),
    2000: (4, 5,4,3,2,6, 45,"Trade growing - 3B USD"),
    2001: (4, 5,4,3,2,6, 44,"SCO formed - new multilateral engagement"),
    2002: (4, 5,4,3,2,6, 44,"Trade reaches 5B USD"),
    2003: (3, 6,3,3,2,7, 38,"Vajpayee visits China - strategic partnership"),
    2004: (3, 6,3,3,2,7, 37,"Trade growing fast"),
    2005: (3, 6,3,4,2,7, 36,"Strategic and cooperative partnership"),
    2006: (4, 5,4,4,3,6, 45,"Arunachal stapled visa controversy"),
    2007: (4, 5,4,4,3,6, 46,"Border tensions increase"),
    2008: (4, 5,4,4,3,6, 46,"Trade reaches 52B USD"),
    2009: (5, 4,5,4,4,5, 54,"Chinese incursions increase"),
    2010: (5, 4,5,4,4,5, 53,"Stapled visa controversy escalates"),
    2011: (5, 4,5,5,4,5, 52,"Trade reaches 74B USD"),
    2012: (5, 4,5,5,5,5, 52,"Depsang standoff"),
    2013: (6, 4,6,5,5,5, 59,"Depsang + Chumar standoffs"),
    2014: (5, 5,5,5,5,6, 50,"Modi-Xi meet - reset attempted"),
    2015: (5, 5,5,5,5,6, 49,"Stable - trade 71B USD"),
    2016: (5, 4,5,5,6,5, 53,"China blocks NSG bid for India"),
    2017: (7, 3,7,6,6,4, 68,"DOKLAM STANDOFF - 73 days"),
    2018: (4, 6,4,5,6,6, 44,"Wuhan informal summit - reset"),
    2019: (5, 5,5,6,7,5, 53,"Article 370 - China objects strongly"),
    2020: (8, 2,9,7,8,3, 78,"GALWAN CLASH - 20 Indian soldiers killed"),
    2021: (7, 3,7,7,8,4, 68,"Partial disengagement at some points"),
    2022: (7, 3,7,7,8,4, 67,"Tawang clash December 2022"),
    2023: (6, 4,6,7,8,5, 60,"Partial disengagement continues"),
    2024: (5, 5,5,7,8,6, 52,"Depsang-Demchok full disengagement"),
}

# Build rows
for year, vals in ip_events.items():
    k, d, l, n, cy, comp, risk, event = vals
    nw_pak   = max(0, (year - 1987) * 7)  if year >= 1987 else 0
    nw_india = max(0, (year - 1974) * 8)  if year >= 1974 else 0
    wars_so_far = sum(1 for w in [1947, 1965, 1971, 1999] if w <= year)
    trade = 1.0 if year < 2000 else (2.0 if year < 2010 else 1.5)

    rows.append({
        'Year': year,
        'Country_Pair': 'India-Pakistan',
        'Military_Ratio': get_ratio(year, ip_mil_ratio),
        'Nuclear_Weapons': nw_pak,
        'India_Nuclear': nw_india,
        'Historical_Wars': wars_so_far,
        'Border_Length_km': 3323,
        'Terrain_Difficulty': 8,
        'Kashmir_Tension': k,
        'LAC_Tension': 0.0,
        'Cyber_Attack_Risk': cy,
        'Trade_Dependency': trade,
        'Diplomatic_Relations': d,
        'Land_Attack_Risk': l,
        'Naval_Attack_Risk': n,
        'Compromise_Chance': comp * 10,
        'Conflict_Risk_Score': risk,
        'Real_Event': event
    })

for year, vals in ic_events.items():
    lac, d, l, n, cy, comp, risk, event = vals
    nw_china = max(0, (year - 1964) * 10) if year >= 1964 else 0
    nw_india = max(0, (year - 1974) * 8)  if year >= 1974 else 0
    trade = max(0, (year - 1990) * 0.5) if year >= 1990 else 0

    rows.append({
        'Year': year,
        'Country_Pair': 'India-China',
        'Military_Ratio': get_ratio(year, ic_mil_ratio),
        'Nuclear_Weapons': nw_china,
        'India_Nuclear': nw_india,
        'Historical_Wars': 1 if year >= 1962 else 0,
        'Border_Length_km': 3488,
        'Terrain_Difficulty': 9,
        'Kashmir_Tension': 0.0,
        'LAC_Tension': lac,
        'Cyber_Attack_Risk': cy,
        'Trade_Dependency': trade,
        'Diplomatic_Relations': d,
        'Land_Attack_Risk': l,
        'Naval_Attack_Risk': n,
        'Compromise_Chance': comp * 10,
        'Conflict_Risk_Score': risk,
        'Real_Event': event
    })

df = pd.DataFrame(rows).sort_values(['Country_Pair', 'Year']).reset_index(drop=True)

def risk_label(s):
    if s <= 50: return 'LOW'
    elif s <= 70: return 'MEDIUM'
    else: return 'HIGH'

df['Risk_Label'] = df['Conflict_Risk_Score'].apply(risk_label)

print(f"Total rows: {len(df)}")
print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
print(f"\nRisk distribution:")
print(df['Risk_Label'].value_counts())
print(f"\nCountry pair:")
print(df['Country_Pair'].value_counts())
print(f"\nWar years (Risk >= 90):")
print(df[df['Conflict_Risk_Score'] >= 90][['Year','Country_Pair','Conflict_Risk_Score','Real_Event']].to_string())

df.to_csv(r"D:\drdo\historical_dataset_1947_2024.csv", index=False)
print("\nSaved to D:\\drdo\\historical_dataset_1947_2024.csv")