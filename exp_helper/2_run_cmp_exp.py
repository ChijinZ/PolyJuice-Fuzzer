import sys
import os
import copy
import shutil
from datetime import datetime

import matplotlib
from matplotlib_venn import venn3_unweighted
import subprocess
import matplotlib.pyplot as plt

from exp_helper.bug_analysis import analyze_crash, analyze_silent

FUZZING_TIME = int(24 * 60 * 60)
ADDITIONAL_WAITING_TIME = 20

TARGET = "tvm"
MODEL_TYPE = "onnx"
OUTPUT_PATH = "/root/2_cmp_exp/"

POLYJUICE_ROOT_PATH = "/root/polyjuice/"
NNSMITH_ROOT_PATH = "/root/nnsmith/"
MTDLCOMP_ROOT_PATH = "/root/MT_DLComp/"

NOISE_REDIRECTION = subprocess.DEVNULL


def get_current_time_str():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def check_all_outputs_normal() -> bool:
    if os.path.exists(os.path.join(OUTPUT_PATH, f"fuzz_report_polyjuice_{TARGET}_0")) or \
            os.path.exists(os.path.join(OUTPUT_PATH, f"fuzz_report_nnsmith_{TARGET}_0")) or \
            os.path.exists(os.path.join(OUTPUT_PATH, f"fuzz_report_mtdlcomp_{TARGET}_0")):
        return False

    polyjuice_output_path = os.path.join(OUTPUT_PATH, f"fuzz_report_polyjuice_{TARGET}_0")
    polyjuice_output_log_0_path = os.path.join(polyjuice_output_path, "fuzz_log_0")
    with open(polyjuice_output_log_0_path, "r") as f:
        if len(f.readlines()) < 100:
            print(f.readlines())
            return False

    nnsmith_output_path = os.path.join(OUTPUT_PATH, f"fuzz_report_nnsmith_{TARGET}_0")
    nnsmith_output_log_0_path = os.path.join(nnsmith_output_path, "fuzz_log_0")
    with open(nnsmith_output_log_0_path, "r") as f:
        if len(f.readlines()) < 100:
            print(f.readlines())
            return False

    return True


def run_polyjuice(fuzzing_time: int) -> subprocess.Popen:
    fuzz_loop_script_path = os.path.join(POLYJUICE_ROOT_PATH, "equality_saturation_helper/fuzz_loop.py")
    fuzz_script_path = os.path.join(POLYJUICE_ROOT_PATH, "nnsmith/cli/equivalent_fuzz.py")
    output_path = os.path.join(OUTPUT_PATH, f"fuzz_report_polyjuice_{TARGET}_0")
    new_env = copy.deepcopy(os.environ)
    new_env["PYTHONPATH"] = "/root/polyjuice"
    new_env["EQ_HELPER_PATH"] = "/root/polyjuice/libdl_compiler_fuzzer_helper.so"
    new_env["SAVE_EQ_IS_WRONG"] = "false"
    p = subprocess.Popen(
        ["python3", fuzz_loop_script_path, "-o", output_path, "-s", fuzz_script_path, "-m", MODEL_TYPE, "-b", TARGET,
         "-p", "1", "-r", str(fuzzing_time), "-t", "cpu"], stdout=NOISE_REDIRECTION, stderr=NOISE_REDIRECTION,
        env=new_env, cwd=POLYJUICE_ROOT_PATH)

    return p


def run_nnsmith(fuzzing_time: int) -> subprocess.Popen:
    fuzz_loop_script_path = os.path.join(POLYJUICE_ROOT_PATH, "equality_saturation_helper/fuzz_loop.py")
    fuzz_script_path = os.path.join(NNSMITH_ROOT_PATH, "nnsmith/cli/fuzz.py")
    output_path = os.path.join(OUTPUT_PATH, f"fuzz_report_nnsmith_{TARGET}_0")
    new_env = copy.deepcopy(os.environ)
    new_env["PYTHONPATH"] = "/root/nnsmith"
    p = subprocess.Popen(
        ["python3", fuzz_loop_script_path, "-o", output_path, "-s", fuzz_script_path, "-m", MODEL_TYPE, "-b", TARGET,
         "-p", "1", "-r", str(fuzzing_time), "-t", "cpu"], stdout=NOISE_REDIRECTION, stderr=NOISE_REDIRECTION,
        env=new_env, cwd=NNSMITH_ROOT_PATH)

    return p


def run_mt_dlcomp(fuzzing_time: int) -> subprocess.Popen:
    fuzzer_dir_path = os.path.join(MTDLCOMP_ROOT_PATH, "Testing-DNN-Compilers")
    fuzz_script_path = os.path.join(fuzzer_dir_path, "fuzz_loop.py")
    p = subprocess.Popen(["python3", fuzz_script_path, "-c", TARGET, "-t", str(fuzzing_time)],
                         stdout=NOISE_REDIRECTION,
                         stderr=NOISE_REDIRECTION, cwd=fuzzer_dir_path)

    return p


def process_results():
    pass


def run_fuzzers(fuzzing_time):
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    mt_dlcomp_tmp_output_path = os.path.join(MTDLCOMP_ROOT_PATH, "bugs")
    if os.path.exists(mt_dlcomp_tmp_output_path):
        shutil.rmtree(mt_dlcomp_tmp_output_path)

    polyjuice_proc = run_polyjuice(fuzzing_time)
    nnsmith_proc = run_nnsmith(fuzzing_time)
    mt_dlcomp_proc = run_mt_dlcomp(fuzzing_time)

    print(f"[{get_current_time_str()}] waiting for all fuzzers to finish")
    try:
        polyjuice_proc.wait(timeout=fuzzing_time + ADDITIONAL_WAITING_TIME)
        print("polyjuice finished")
    except subprocess.TimeoutExpired:
        polyjuice_proc.kill()
        print("polyjuice killed because of timeout")
    except KeyboardInterrupt:
        print("keyboard interrupt")
        polyjuice_proc.kill()
    try:
        nnsmith_proc.wait(timeout=fuzzing_time + ADDITIONAL_WAITING_TIME)
        print("nnsmith finished")
    except subprocess.TimeoutExpired:
        nnsmith_proc.kill()
        print("nnsmith killed because of timeout")
    except KeyboardInterrupt:
        print("keyboard interrupt")
        nnsmith_proc.kill()
    try:
        mt_dlcomp_proc.wait(timeout=fuzzing_time + ADDITIONAL_WAITING_TIME)
        print("mt_dlcomp finished")
    except subprocess.TimeoutExpired:
        mt_dlcomp_proc.kill()
        print("mt_dlcomp killed because of timeout")
    except KeyboardInterrupt:
        print("keyboard interrupt")
        mt_dlcomp_proc.kill()
    print(f"[{get_current_time_str()}] all fuzzers finished")

    mt_dlcomp_output_path = os.path.join(OUTPUT_PATH, f"fuzz_report_mtdlcomp_{TARGET}_0")
    shutil.copytree(mt_dlcomp_tmp_output_path, mt_dlcomp_output_path)

    polyjuice_output_path = os.path.join(OUTPUT_PATH, f"fuzz_report_polyjuice_{TARGET}_0")
    nnsmith_output_path = os.path.join(OUTPUT_PATH, f"fuzz_report_nnsmith_{TARGET}_0")

    print(f"[{get_current_time_str()}] checking if all outputs are normal")
    if check_all_outputs_normal():
        print(f"[{get_current_time_str()}] all outputs seem normal")
    else:
        print(f"[{get_current_time_str()}] some outputs are abnormal")

    print(f"[{get_current_time_str()}] start to process results")
    polyjuice_tvm_crash_set = analyze_crash(polyjuice_output_path)
    nnsmith_tvm_crash_set = analyze_crash(nnsmith_output_path)
    mtdlcomp_tvm_crash_set = analyze_crash(mt_dlcomp_output_path)

    polyjuice_tvm_diff_set, polyjuice_tvm_eq_diff_set = analyze_silent(polyjuice_output_path)
    nnsmith_tvm_diff_set, nnsmith_tvm_eq_diff_set = analyze_silent(nnsmith_output_path)
    mtdlcomp_tvm_diff_set, mtdlcomp_tvm_eq_diff_set = analyze_silent(mt_dlcomp_output_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    # order: NNSmith, MT-DLComp, PolyJuice
    v = venn3_unweighted(subsets=({
        '100': len(
            nnsmith_tvm_crash_set - (nnsmith_tvm_crash_set & mtdlcomp_tvm_crash_set & polyjuice_tvm_crash_set) - (
                    nnsmith_tvm_crash_set & mtdlcomp_tvm_crash_set) - (
                    nnsmith_tvm_crash_set & polyjuice_tvm_crash_set)
        ),
        '001': len(
            polyjuice_tvm_crash_set - (nnsmith_tvm_crash_set & mtdlcomp_tvm_crash_set & polyjuice_tvm_crash_set) - (
                    polyjuice_tvm_crash_set & mtdlcomp_tvm_crash_set) - (
                    polyjuice_tvm_crash_set & nnsmith_tvm_crash_set)
        ),
        '010': len(
            mtdlcomp_tvm_crash_set - (nnsmith_tvm_crash_set & mtdlcomp_tvm_crash_set & polyjuice_tvm_crash_set) - (
                    mtdlcomp_tvm_crash_set & polyjuice_tvm_crash_set) - (
                    mtdlcomp_tvm_crash_set & nnsmith_tvm_crash_set)
        ),
        '110': len(nnsmith_tvm_crash_set & mtdlcomp_tvm_crash_set - (
                nnsmith_tvm_crash_set & mtdlcomp_tvm_crash_set & polyjuice_tvm_crash_set)),
        '101': len(nnsmith_tvm_crash_set & polyjuice_tvm_crash_set - (
                nnsmith_tvm_crash_set & mtdlcomp_tvm_crash_set & polyjuice_tvm_crash_set)),
        '011': len(mtdlcomp_tvm_crash_set & polyjuice_tvm_crash_set - (
                nnsmith_tvm_crash_set & mtdlcomp_tvm_crash_set & polyjuice_tvm_crash_set)),
        '111': len(nnsmith_tvm_crash_set & mtdlcomp_tvm_crash_set & polyjuice_tvm_crash_set)
    }), set_labels=('NNSmith', 'MT-DLComp', 'PolyJuice'),
        set_colors=("dimgray", "sienna", "cadetblue"), ax=axes[0][0])
    axes[0][0].set_title("TVM Crash/Exception Bugs", loc="center", fontdict={'fontsize': 30})
    for text in v.set_labels:
        text.set_fontsize(20)
    for text in v.subset_labels:
        text.set_fontsize(25)

    v = venn3_unweighted(subsets=({
        '100': len(nnsmith_tvm_diff_set),
        '001': len(polyjuice_tvm_eq_diff_set),
        '010': len(mtdlcomp_tvm_diff_set),
        '110': len(nnsmith_tvm_diff_set & mtdlcomp_tvm_diff_set),
        '101': len(nnsmith_tvm_diff_set & polyjuice_tvm_diff_set),
        '011': len(mtdlcomp_tvm_diff_set & polyjuice_tvm_diff_set),
        '111': len(nnsmith_tvm_diff_set & mtdlcomp_tvm_diff_set & polyjuice_tvm_diff_set)
    }), set_labels=('NNSmith', 'MT-DLComp', 'PolyJuice'),
        set_colors=("dimgray", "sienna", "cadetblue"), ax=axes[1][0])
    axes[1][0].set_title("TVM Mis-Compilation Bugs", loc="center", fontdict={'fontsize': 30})
    for text in v.set_labels:
        text.set_fontsize(20)
    for text in v.subset_labels:
        text.set_fontsize(25)

    figure_fig = plt.gcf()  # 'get current figure'

    output_path = os.path.join(OUTPUT_PATH, "fig10.pdf")
    if os.path.exists(output_path):
        os.remove(output_path)
    figure_fig.savefig(output_path,
                       format='pdf',
                       dpi=1000,
                       bbox_inches='tight',
                       pad_inches=0)


if __name__ == '__main__':
    run_fuzzers(FUZZING_TIME)
