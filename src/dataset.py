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
        print(f"🔍 Scanning {self.data_dir} (SNIPER MODE: clean/1m.csv)...")
        target_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                # SNIPER FILTER: Only allow 'clean/1m.csv'
                if file != "1m.csv":
                    continue
                
                # Check parent folder is 'clean'
                if os.path.basename(root) != "clean":
                    continue
                    
                target_files.append(os.path.join(root, file))
                
        target_files.sort() # Ensure deterministic order
        print(f"📝 Found {len(target_files)} CLEAN 1m CSV files.")
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

    def tokenize_and_save(self, files, limit=None):
        bin_path = os.path.join(self.output_dir, "bins.npy")
        if not hasattr(self, 'bins') or not self.bins:
            if os.path.exists(bin_path):
                self.bins = np.load(bin_path, allow_pickle=True).item()
            else:
                print("❌ Quantiles not found.")
                return

        print(f"🍱 Tokenizing V4 Solid Split (File-Based, Limit={limit})...")
        
        # 1. Random Selection
        np.random.seed(42)
        if limit and len(files) > limit:
            selected_files = np.random.choice(files, limit, replace=False)
        else:
            selected_files = files
            
        print(f"📝 Selected {len(selected_files)} files for SOLID verification.")
        
        # 2. File-Based Split (70/30)
        split_idx = int(len(selected_files) * 0.7)
        train_files = selected_files[:split_idx]
        val_files = selected_files[split_idx:]
        
        print(f"📊 Train Files: {len(train_files)} (e.g. {os.path.basename(train_files[0])})")
        print(f"📊 Val Files:   {len(val_files)}   (e.g. {os.path.basename(val_files[0])})")

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
        
        def process_set(file_list, file_handle, split_name):
            total_tokens = 0
            count = 0
            for f in tqdm(file_list, desc=f"Tokenizing {split_name}", unit="file"):
                feat = self.process_file(f)
                if feat is None: continue
                
                # Debug Verification: Show Relative Path (Ticker/clean/1m.csv)
                # Assuming data structure: .../Ticker/clean/1m.csv
                # We can try to grab the last 3 parts of the path
                rel_parts = f.replace("\\", "/").split("/")[-3:]
                rel_name = "/".join(rel_parts)
                
                if count < 5: # Print first 5 to be sure
                     print(f"   [VERIFY {split_name}] {rel_name}: {feat.index[0]} -> {feat.index[-1]}")
                
                t_dow = (feat['dow'].values + OFFSET_DOW).astype(np.uint16)
                t_hour = (feat['hour'].values + OFFSET_HOUR).astype(np.uint16)
                
                def quantize(values, bins, offset):
                    inds = np.digitize(values, bins) 
                    return (inds + offset).astype(np.uint16)
                
                t_volat = quantize(feat['volat_lag'].values, self.bins['volat_lag'], OFFSET_VOLAT)
                t_vol = quantize(feat['vol_lag'].values, self.bins['vol_lag'], OFFSET_VOL)
                t_ret = quantize(feat['log_ret'].values, self.bins['log_ret'], OFFSET_RET)
                
                stack = np.vstack([t_dow, t_hour, t_volat, t_vol, t_ret])
                flat = stack.T.flatten()
                file_handle.write(flat.tobytes())
                total_tokens += len(flat)
                count += 1
            return total_tokens

        total_toks_train = process_set(train_files, f_train, "TRAIN")
        total_toks_val = process_set(val_files, f_val, "VAL")
            
        f_train.close()
        f_val.close()
        
        print(f"📊 Train Tokens: {total_toks_train}")
        print(f"📊 Val Tokens:   {total_toks_val}")
        print(f"✅ Data Preparation Complete.")
        print(f"📊 Train Tokens: {total_toks_train:,}")
        print(f"📊 Val Tokens:   {total_toks_val:,}")
        print(f"💾 Saved to {self.output_dir}/")
