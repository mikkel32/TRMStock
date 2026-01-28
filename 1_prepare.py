"""
1_prepare.py
Step 1: Data Preparation.

This script scans your 'StockData' folder, learns the market statistics,
and converts everything into a causal token stream for the TRM.

Usage:
    python 1_prepare.py
"""

from src.dataset import DataEngine
import os

# --- Settings ---
# You can change these paths if you move folders
INPUT_FOLDER = os.getenv("TRM_INPUT_DATA", "StockData")
OUTPUT_FOLDER = os.getenv("TRM_OUTPUT_DATA", "data")

def main():
    print("🔹 ExperimentLM V4: Data Preparation")
    print(f"   Input:  {INPUT_FOLDER}")
    print(f"   Output: {OUTPUT_FOLDER}")
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    engine = DataEngine(INPUT_FOLDER, OUTPUT_FOLDER)
    files = engine.scan_files()
    
    if not files:
        print("❌ No CSV files found. Check your INPUT_FOLDER path.")
        return
        
    # Phase 1: Learn Statistics (Quantiles)
    engine.learn_quantiles(files)
    
    # Phase 2: Create Binary Datasets
    engine.tokenize_and_save(files)

if __name__ == "__main__":
    main()
