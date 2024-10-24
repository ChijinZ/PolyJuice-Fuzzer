import re
import pandas as pd
import sys
import os

TIME_LOG_DIR_PATH = "/root/raw_data/raw_data/tab5_raw_data"
OUTPUT_DIR_PATH = "/root/raw_data/raw_data_results"


def print_avg_time(file_path):
    pattern = re.compile(
        r"Timing: gen: \s*(-?\d+\.\d+?)ms, saturation: \s*(-?\d+\.\d+?)ms, egraph_gen: \s*(-?\d+\.\d+?)ms, eval: \s*(-?\d+\.\d+?)ms,")
    record = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            match = pattern.search(line)
            if match is not None:
                assert len(match.groups()) == 4
                tmp = []
                for i in range(4):
                    tmp.append(float(match.group(i + 1)))
                record.append(tmp)
    df = pd.DataFrame(data=record, columns=["gen", "rewriting", "extraction", "execution"])
    return df


def main():
    df_xla_5 = print_avg_time(os.path.join(TIME_LOG_DIR_PATH, "./time_log_xla_5"))
    df_xla_10 = print_avg_time(os.path.join(TIME_LOG_DIR_PATH, "./time_log_xla_10"))
    df_xla_15 = print_avg_time(os.path.join(TIME_LOG_DIR_PATH, "./time_log_xla_15"))
    df_tvm_5 = print_avg_time(os.path.join(TIME_LOG_DIR_PATH, "./time_log_tvm_5"))
    df_tvm_10 = print_avg_time(os.path.join(TIME_LOG_DIR_PATH, "./time_log_tvm_10"))
    df_tvm_15 = print_avg_time(os.path.join(TIME_LOG_DIR_PATH, "./time_log_tvm_15"))

    df = pd.DataFrame(columns=["compiler", "node_num", "gen", "rewriting", "extraction", "execution"])

    df = df.append(
        {"compiler": "tvm", "node_num": 5, "gen": df_tvm_5["gen"].mean(), "rewriting": df_tvm_5["rewriting"].mean(),
         "extraction": df_tvm_5["extraction"].mean(), "execution": df_tvm_5["execution"].mean()}, ignore_index=True)
    df = df.append(
        {"compiler": "tvm", "node_num": 10, "gen": df_tvm_10["gen"].mean(), "rewriting": df_tvm_10["rewriting"].mean(),
         "extraction": df_tvm_10["extraction"].mean(), "execution": df_tvm_10["execution"].mean()}, ignore_index=True)
    df = df.append(
        {"compiler": "tvm", "node_num": 15, "gen": df_tvm_15["gen"].mean(), "rewriting": df_tvm_15["rewriting"].mean(),
         "extraction": df_tvm_15["extraction"].mean(), "execution": df_tvm_15["execution"].mean()}, ignore_index=True)
    df = df.append(
        {"compiler": "xla", "node_num": 5, "gen": df_xla_5["gen"].mean(), "rewriting": df_xla_5["rewriting"].mean(),
         "extraction": df_xla_5["extraction"].mean(), "execution": df_xla_5["execution"].mean()}, ignore_index=True)
    df = df.append(
        {"compiler": "xla", "node_num": 10, "gen": df_xla_10["gen"].mean(), "rewriting": df_xla_10["rewriting"].mean(),
         "extraction": df_xla_10["extraction"].mean(), "execution": df_xla_10["execution"].mean()}, ignore_index=True)
    df = df.append(
        {"compiler": "xla", "node_num": 15, "gen": df_xla_15["gen"].mean(), "rewriting": df_xla_15["rewriting"].mean(),
         "extraction": df_xla_15["extraction"].mean(), "execution": df_xla_15["execution"].mean()}, ignore_index=True)

    print(df)

    output_path = os.path.join(OUTPUT_DIR_PATH, "tab5.csv")
    if os.path.exists(output_path):
        os.remove(output_path)
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()
