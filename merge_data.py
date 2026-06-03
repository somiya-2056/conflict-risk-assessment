import pandas as pd

ucdp = pd.read_csv(r"D:\drdo\GEDEvent_v25_1.csv", low_memory=False)
ml_df = pd.read_csv(r"D:\drdo\ml_dataset.csv")

mask = ucdp['country'].isin(['India', 'Pakistan', 'China'])
ucdp_f = ucdp[mask]

yearly = ucdp_f.groupby(['country', 'year']).agg(
    conflict_events=('id', 'count'),
    total_deaths=('best', 'sum')
).reset_index()

def get_conflict_data(row, country, yearly_data):
    year = row['Year']
    data = yearly_data[(yearly_data['country'] == country) & 
                       (yearly_data['year'] == year)]
    if len(data) > 0:
        return data['conflict_events'].values[0], data['total_deaths'].values[0]
    return 0, 0

india_events, india_deaths, enemy_events, enemy_deaths = [], [], [], []

for _, row in ml_df.iterrows():
    ie, id_ = get_conflict_data(row, 'India', yearly)
    enemy = 'Pakistan' if row['Country_Pair'] == 'India-Pakistan' else 'China'
    ee, ed = get_conflict_data(row, enemy, yearly)
    india_events.append(ie)
    india_deaths.append(id_)
    enemy_events.append(ee)
    enemy_deaths.append(ed)

ml_df['India_Conflict_Events'] = india_events
ml_df['India_Deaths'] = india_deaths
ml_df['Enemy_Conflict_Events'] = enemy_events
ml_df['Enemy_Deaths'] = enemy_deaths

ml_df.to_csv(r"D:\drdo\ml_dataset_v2.csv", index=False)
print("Dataset updated! Total rows:", len(ml_df))
print(ml_df[['Year','Country_Pair','India_Conflict_Events','Enemy_Conflict_Events','Enemy_Deaths']].head(10))