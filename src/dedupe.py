"""Uklanja duple kombinacije iz CSV rezultata.

Ključ retka čine scenario, task, reprezentacija, odabir kanala i klasifikator.
Ako je skripta prekinuta i ista kombinacija kasnije ponovno izračunata, ostavlja
se zadnji redak jer je to obično rezultat zadnjeg pokretanja. Pokretanje:
python dedupe.py putanja_do.csv
"""
import sys
import pandas as pd
 
path = sys.argv[1] if len(sys.argv) > 1 else 'results/experiment_cross_subject.csv'
df = pd.read_csv(path)
key_cols = ['scenario', 'task', 'feature_type', 'channel_selection', 'classifier']
 
n_before = len(df)
df = df.drop_duplicates(subset=key_cols, keep='last')
n_after = len(df)
 
df.to_csv(path, index=False)
print(f"{path}: {n_before} -> {n_after} redaka ({n_before - n_after} duplikata uklonjeno)")
