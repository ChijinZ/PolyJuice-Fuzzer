import sys
import os

from exp_helper.process_diff_effect import process_diff_effect

RAW_DATA_PATH = "/root/raw_data/raw_data/tab4_raw_data/ablation_exp_5000.csv"
OUTPUT_DIR_PATH = "/root/raw_data/raw_data_results"


def main():
    os.environ["OUTPUT_DIR"] = OUTPUT_DIR_PATH
    process_diff_effect(RAW_DATA_PATH)


if __name__ == '__main__':
    main()
