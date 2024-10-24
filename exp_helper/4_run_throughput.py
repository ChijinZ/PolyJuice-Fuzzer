import subprocess
import shutil
import os
import copy
import re
import pandas as pd

POLYJUICE_ROOT_PATH = "/root/polyjuice/"
OUTPUT_DIR = "/root/4_throughput/"

TMP_DIR = "/tmp/throughput/"

MAX_TIME = 2 * 60 * 60


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
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR, exist_ok=True)

    new_env = copy.deepcopy(os.environ)
    new_env["PYTHONPATH"] = "/root/polyjuice"
    new_env["EQ_HELPER_PATH"] = "/root/polyjuice/libdl_compiler_fuzzer_helper.so"
    new_env["SAVE_EQ_IS_WRONG"] = "false"
    new_env["TIME_EXP_RUN"] = "true"

    tvm_5_log_path = f"{OUTPUT_DIR}/time_log_tvm_5"
    tvm_10_log_path = f"{OUTPUT_DIR}/time_log_tvm_10"

    print("start running")
    tvm_5_f = open(tvm_5_log_path, "w")
    tvm_10_f = open(tvm_10_log_path, "w")
    tvm_5_node_proc = subprocess.Popen(
        ["python3", f"{POLYJUICE_ROOT_PATH}/nnsmith/cli/equivalent_fuzz.py", f"fuzz.time={MAX_TIME}",
         "model.type=onnx",
         "backend.type=tvm",
         "backend.target=cpu", f"fuzz.root={TMP_DIR}/tvm_5", "debug.viz=true", "mgen.max_nodes=5"], env=new_env,
        cwd=POLYJUICE_ROOT_PATH, stdout=tvm_5_f, stderr=tvm_5_f)
    tvm_10_node_proc = subprocess.Popen(
        ["python3", f"{POLYJUICE_ROOT_PATH}/nnsmith/cli/equivalent_fuzz.py", f"fuzz.time={MAX_TIME}",
         "model.type=onnx",
         "backend.type=tvm",
         "backend.target=cpu", f"fuzz.root={TMP_DIR}/tvm_10", "debug.viz=true", "mgen.max_nodes=10"],
        env=new_env,
        cwd=POLYJUICE_ROOT_PATH, stdout=tvm_10_f, stderr=tvm_10_f)
    try:
        tvm_5_node_proc.wait(timeout=MAX_TIME + 60)
    except subprocess.TimeoutExpired:
        tvm_5_node_proc.kill()
        print("tvm_5 killed because of timeout")

    try:
        tvm_10_node_proc.wait(timeout=MAX_TIME + 60)
    except subprocess.TimeoutExpired:
        tvm_10_node_proc.kill()
        print("tvm_10 killed because of timeout")

    tvm_5_f.close()
    tvm_10_f.close()

    print("finished running")

    df_tvm_5 = print_avg_time(tvm_5_log_path)
    df_tvm_10 = print_avg_time(tvm_10_log_path)

    df = pd.DataFrame(columns=["compiler", "node_num", "gen", "rewriting", "extraction", "execution"])

    df = df.append(
        {"compiler": "tvm", "node_num": 5, "gen": df_tvm_5["gen"].mean(), "rewriting": df_tvm_5["rewriting"].mean(),
         "extraction": df_tvm_5["extraction"].mean(), "execution": df_tvm_5["execution"].mean()}, ignore_index=True)
    df = df.append(
        {"compiler": "tvm", "node_num": 10, "gen": df_tvm_10["gen"].mean(), "rewriting": df_tvm_10["rewriting"].mean(),
         "extraction": df_tvm_10["extraction"].mean(), "execution": df_tvm_10["execution"].mean()}, ignore_index=True)

    print(df)
    df.to_csv(f"{OUTPUT_DIR}/tab5.csv", index=False)


if __name__ == '__main__':
    main()
