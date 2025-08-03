import numpy as np
import pandas as pd
import re
import json
import glob

def extract_date_set(files_json):
    dates = set()
    with open(files_json) as f:
        for fn in json.load(f):
            m = re.search(r"(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})", fn)
            if m:
                start, end = m.groups()
                dates |= set(pd.date_range(start, end).strftime("%Y-%m-%d"))
    return dates

def load_memmap(path):
    arr = np.load(path, mmap_mode='r')
    return np.memmap(path, dtype=arr.dtype, mode='r', shape=arr.shape)

def parse_key(key):
    date = key[:10]
    rest = key[11:]
    i = rest.rfind('_')
    return date, rest[:i], rest[i+1:]

def build_ticker_industry_map(pattern, chunksize):
    mp = {}
    for fn in glob.glob(pattern):
        for chunk in pd.read_csv(fn, usecols=['ticker','gics'], dtype=str, chunksize=chunksize):
            chunk = chunk.dropna(subset=['ticker','gics'])
            for t, g in zip(chunk['ticker'], chunk['gics']):
                mp.setdefault(t, g)
    return mp

def parse_gics_hierarchy(block):
    """
    Regex‐based: find all (code, name) pairs, then bucket by code length.
    """
    sector_map, group_map, industry_map, subind_map = {}, {}, {}, {}
    # capture code (2–8 digits) + space(s) + name (up to next code or line end)
    pairs = re.findall(r'(\d{2,8})\s+([A-Za-z& ,\-\(\)]+)', block)
    for code, name in pairs:
        name = name.strip()
        if len(code) == 2:
            sector_map[code] = name
        elif len(code) == 4:
            group_map[code] = name
        elif len(code) == 6:
            industry_map[code] = name
        elif len(code) == 8:
            subind_map[code] = name
    return sector_map, group_map, industry_map, subind_map

def parse_date(key):
    return pd.to_datetime(key[:10])

def build_ticker_gics_map(pattern, chunksize):
    mp = {}
    for fn in glob.glob(pattern):
        for chunk in pd.read_csv(fn, usecols=['ticker','gics'], dtype=str, chunksize=chunksize):
            chunk = chunk.dropna(subset=['ticker','gics'])
            for t, g in zip(chunk['ticker'], chunk['gics']):
                mp.setdefault(t, g)
    return mp

def sign(x):
    return np.where(x>0, 1, np.where(x<0, -1, 0))