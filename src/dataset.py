"""
src/dataset.py
The V4 Causal Data Engine.

Internal logic for scanning, processing, and tokenizing financial data.
Imported by 1_prepare.py.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

VOCAB_SIZE = 8192
MIN_ROWS = 1000

class DataEngine:
    def __init__(self, data_dir, output_dir, vocab_size=VOCAB_SIZE):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.vocab_size = vocab_size
        self.bins = {} 

    def scan_files(self):
        print(f"🔍 Scanning {self.data_dir}...")
        target_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".csv"):
                    target_files.append(os.path.join(root, file))
        target_files.sort() # Ensure deterministic order
        print(f"📝 Found {len(target_files)} CSV files.")
        return target_files

    def process_file(self, file_path):
        """
        Returns DataFrame with columns: ['dow', 'hour', 'volat_lag', 'vol_lag', 'log_ret']
        """
        try:
            df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
            df.sort_index(inplace=True) # Guarantee chronological order (V4 Fix for Leak)
            
            # Debug: Verify Order for User
            if "AAPL.csv" in file_path or "NVDA.csv" in file_path:
                print(f"\n[VERIFY] {os.path.basename(file_path)}: {df.index[0]} -> {df.index[-1]}")
                print(f"         Sorted? {df.index.is_monotonic_increasing}")
            
            # Map columns
            cols = {c.lower(): c for c in df.columns}
            if 'adj close' in cols: close_col = cols['adj close']
            elif 'close' in cols: close_col = cols['close']
            else: return None
                
            required = ['open', 'high', 'low', 'volume']
            if not all(c in [k.lower() for k in df.columns] for c in required): return None
            
            df = df.rename(columns={
                cols.get('open', 'open'): 'Open',
                cols.get('high', 'high'): 'High',
                cols.get('low', 'low'): 'Low',
                cols.get('volume', 'volume'): 'Volume',
                close_col: 'Close'
            })
            
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            if len(df) < MIN_ROWS: return None
            
            # 1. Base Features
            df['dow'] = df.index.dayofweek
            df['hour'] = df.index.hour
            df['volat_t'] = (df['High'] - df['Low']) / (df['Open'].replace(0, np.nan))
            df['log_vol_t'] = np.log(df['Volume'] + 1)
            df['log_ret'] = np.diff(np.log(df['Close']), prepend=np.nan)
            
            # 2. Causal Shift (V4 Fix)
            # Predictors from t-1 predict t
            df['volat_lag'] = df['volat_t'].shift(1)
            df['vol_lag'] = df['log_vol_t'].shift(1)
            
            df = df.dropna()
            
            features = df[['dow', 'hour', 'volat_lag', 'vol_lag', 'log_ret']].copy()
            
            # Filter Extreme Outliers
            mask = np.abs(features['log_ret']) < 0.4
            features = features[mask]
            
            if len(features) < MIN_ROWS: return None
            
            return features
            
        except Exception:
            return None

    def learn_quantiles(self, files):
        print("📊 Learning quantiles (bins)...")
        reservoir = {'volat_lag': [], 'vol_lag': [], 'log_ret': []}
        
        np.random.seed(42)
        sample_files = np.random.choice(files, min(len(files), 200), replace=False)
        
        for f in tqdm(sample_files, desc="Sampling", unit="file"):
            feat = self.process_file(f)
            if feat is not None:
                for col in reservoir:
                    reservoir[col].extend(feat[col].values[:5000])
        
        self.bins = {}
        target_bins = 2000 
        q_steps = np.linspace(0, 100, target_bins + 1)
        
        for col, data in reservoir.items():
            if not data: continue
            arr = np.array(data)
            arr = arr[np.isfinite(arr)]
            thresholds = np.percentile(arr, q_steps)
            self.bins[col] = np.unique(thresholds)
            
        np.save(os.path.join(self.output_dir, "bins.npy"), self.bins)

    def tokenize_and_save(self, files):
        bin_path = os.path.join(self.output_dir, "bins.npy")
        if not hasattr(self, 'bins') or not self.bins:
            if os.path.exists(bin_path):
                self.bins = np.load(bin_path, allow_pickle=True).item()
            else:
                print("❌ Quantiles not found.")
                return

        print("🍱 Tokenizing V4 Causal Stream (Strict Time Split)...")
        # Ensure cutoff is UTC to match the data.
        # Data found to start ~Dec 2024. Using Oct 2025 as split (~80/20).
        CUTOFF_DATE = pd.Timestamp("2025-10-01").tz_localize("UTC")
        print(f"📅 Split Date: {CUTOFF_DATE}")
        
        # Schema: [DOW, Hour, Volat(t-1), Vol(t-1), Ret(t)]
        OFFSET_DOW = 0
        OFFSET_HOUR = 7
        OFFSET_VOLAT = 31
        OFFSET_VOL = OFFSET_VOLAT + len(self.bins['volat_lag']) + 1
        OFFSET_RET = OFFSET_VOL + len(self.bins['vol_lag']) + 1
        
        max_token_id = OFFSET_RET + len(self.bins['log_ret']) + 1
        print(f"🔢 Max Token ID: {max_token_id}")
        
        train_path = os.path.join(self.output_dir, "train.bin")
        val_path = os.path.join(self.output_dir, "val.bin")
        
        f_train = open(train_path, "wb")
        f_val = open(val_path, "wb")
        
        total_toks_train = 0
        total_toks_val = 0
        
        for f in tqdm(files, desc="Tokenizing", unit="file"):
            feat = self.process_file(f)
            if feat is None: continue
            
            # Row-level Time Split
            mask_val = feat.index >= CUTOFF_DATE
            mask_train = ~mask_val
            
            def process_subset(subset, file_handle):
                if len(subset) == 0: return 0
                
                t_dow = (subset['dow'].values + OFFSET_DOW).astype(np.uint16)
                t_hour = (subset['hour'].values + OFFSET_HOUR).astype(np.uint16)
                
                def quantize(values, bins, offset):
                    inds = np.digitize(values, bins) 
                    return (inds + offset).astype(np.uint16)
                
                t_volat = quantize(subset['volat_lag'].values, self.bins['volat_lag'], OFFSET_VOLAT)
                t_vol = quantize(subset['vol_lag'].values, self.bins['vol_lag'], OFFSET_VOL)
                t_ret = quantize(subset['log_ret'].values, self.bins['log_ret'], OFFSET_RET)
                
                stack = np.vstack([t_dow, t_hour, t_volat, t_vol, t_ret])
                flat = stack.T.flatten()
                file_handle.write(flat.tobytes())
                return len(flat)

            total_toks_train += process_subset(feat[mask_train], f_train)
            total_toks_val += process_subset(feat[mask_val], f_val)
            
        f_train.close()
        f_val.close()
        print(f"✅ Data Preparation Complete.")
        print(f"📊 Train Tokens: {total_toks_train:,}")
        print(f"📊 Val Tokens:   {total_toks_val:,}")
        print(f"💾 Saved to {self.output_dir}/")
