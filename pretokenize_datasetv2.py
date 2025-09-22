#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd, torch
from transformers import AutoTokenizer

LABELS = ["bearish","neutral","bullish"]
DATA_FILE = "btc_events_labeled_multi.csv"
TEXT_COLS = ["titles_joined","body_concat"]
TIME_COL = "event_time"
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 256
VAL_RATIO = 0.2

def concat_text(row):
    return " ".join(str(row[c]) for c in TEXT_COLS if c in row and pd.notna(row[c]))

def split(df):
    if TIME_COL in df.columns:
        df = df.sort_values(TIME_COL).reset_index(drop=True)
        cut = int(len(df)*(1-VAL_RATIO))
        return df.iloc[:cut], df.iloc[cut:]
    cut = int(len(df)*(1-VAL_RATIO))
    return df.iloc[:cut], df.iloc[cut:]

df = pd.read_csv(DATA_FILE)
for c in ["label_30m","label_60m","label_120m"]:
    df = df[df[c].isin(LABELS)]
train, val = split(df)
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
def build(df):
    texts = [concat_text(r) for _,r in df.iterrows()]
    enc = tok(texts,truncation=True,padding="max_length",max_length=MAX_LEN,return_tensors="pt")
    labs = {c: torch.tensor([LABELS.index(v) for v in df[c]],dtype=torch.long) for c in ["label_30m","label_60m","label_120m"]}
    return {"encodings":enc,"labels":labs}
torch.save(build(train),"token_cache_train.pt")
torch.save(build(val),"token_cache_val.pt")
print("[ok] Caches écrits : train=",len(train)," val=",len(val))
