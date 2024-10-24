from nnsmith.cli.equivalent_fuzz import EquivalentFuzzingLoop
from nnsmith.cli.fuzz import *
from nnsmith.materialize import BugReport
from typing import List, Tuple
from equality_saturation_helper.helper import EquivalentGraphHelper, FFIHelper, UnreachableException
import re
import difflib
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os
import subprocess
import shutil

MAX_ITERATION = 5000
# MAX_ITERATION = 5
MAX_WORKERS = 15
TESTED_BACKEND = "tvm"

TMP_FILE = "/tmp/ablation_log"


def clean_tmp_files(tmp_path):
    p = subprocess.run(["rm", "-rf", f"{tmp_path}*"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p.returncode == 0


def process_log(log):
    new_log = []
    for line in log:
        if line.startswith("["):
            pattern = r'\[\d{2}:\d{2}:\d{2}\]\s'  # remove time stamp such as [12:34:56]
            log_without_timestamp = re.sub(pattern, '', line)
            pattern_2 = r"(id=)[0-9a-fA-F]+"  # remove id such as id=0x12345678
            log_without_timestamp = re.sub(pattern_2, r"\1", log_without_timestamp)

            if "Cannot emit debug location for undefined span" not in log_without_timestamp \
                    and "Warning" not in log_without_timestamp \
                    and "naive_allocator" not in log_without_timestamp:
                new_log.append(log_without_timestamp)

    return new_log


def edit_distance(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = [[0 for _ in range(size_y)] for _ in range(size_x)]

    for x in range(size_x):
        matrix[x][0] = x
    for y in range(size_y):
        matrix[0][y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x][y] = min(
                    matrix[x - 1][y] + 1,
                    matrix[x - 1][y - 1],
                    matrix[x][y - 1] + 1
                )
            else:
                matrix[x][y] = min(
                    matrix[x - 1][y] + 1,
                    matrix[x - 1][y - 1] + 1,
                    matrix[x][y - 1] + 1
                )
    return matrix[size_x - 1][size_y - 1]


def lcs_distance(log1_lines, log2_lines) -> int:
    # Determine which log is longer
    long_log, short_log = (log1_lines, log2_lines) if len(log1_lines) > len(log2_lines) else (log2_lines, log1_lines)

    diff = difflib.unified_diff(long_log, short_log)

    # diff_lines = len([l for l in diff if l.startswith('-')])
    diff_lines = len(list(diff))

    return diff_lines


def make_test_case_again(testcase: TestCase):
    return TestCase(testcase.model,
                    testcase.model.make_oracle(testcase.oracle.input))


def run(fuzzer: EquivalentFuzzingLoop, testcase: TestCase, thread_id: int) -> List[str]:
    tmp_file = f"{TMP_FILE}_{thread_id}"
    res = fuzzer.execute_testcase(testcase, crash_safe=True, file_path_for_subprocess_log=tmp_file, timeout=20)
    if isinstance(res, BugReport):
        print(res)
        return []

    with open(tmp_file, "r") as f:
        original_log = f.readlines()

    os.remove(tmp_file)

    return original_log


def run_and_get_diff_res(cfg: DictConfig, max_iteration_num: int, thread_id: int) -> List[
    Tuple[int, int, int, int, int, int, int, int]]:
    cfg["fuzz"]["root"] = cfg["fuzz"]["root"] + f"_{thread_id}"
    shutil.rmtree(cfg["fuzz"]["root"], ignore_errors=True)
    fuzzer = EquivalentFuzzingLoop(cfg)
    ffi_helper = FFIHelper()
    assert ffi_helper.is_helper_loaded()

    iteration = 0

    res_record = []

    while iteration < max_iteration_num:
        print(iteration)
        iteration += 1
        seed = random.getrandbits(32)
        # seed = 3850036226
        need_make_ir = True
        while need_make_ir:
            try:
                ir = fuzzer.make_gir(seed)
                need_make_ir = False
            except BaseException as e:
                seed = random.getrandbits(32)
        print(f"seed: {seed}")
        eq_graph_helper = EquivalentGraphHelper(ffi_helper=ffi_helper, gir=ir)

        eq_graph_helper.initialize_inner_graph()
        eq_graph_helper.run_saturation()
        randomly_ir1, graph_output_map1 = eq_graph_helper.randomly_generate_an_equivalent_graph()
        randomly_ir2, graph_output_map2 = eq_graph_helper.randomly_generate_an_equivalent_graph()
        simplified_ir, graph_output_map3 = eq_graph_helper.find_the_most_simplified_equivalent_graph()
        complex_ir, graph_output_map4 = eq_graph_helper.find_the_most_complex_equivalent_graph()
        print("finish ir generation")

        try:
            original_testcase = fuzzer.gir_to_testcase(ir)
            random_testcase1 = fuzzer.gir_to_testcase(randomly_ir1)
            random_testcase2 = fuzzer.gir_to_testcase(randomly_ir2)
            simplified_testcase = fuzzer.gir_to_testcase(simplified_ir)
            complex_testcase = fuzzer.gir_to_testcase(complex_ir)
        except BaseException as e:
            print(f"fail to convert gir to testcase: {e}")
            iteration -= 1
            continue

        try:
            fuzzer.make_inputs_and_weights_consist(original_testcase, random_testcase1)
            fuzzer.make_inputs_and_weights_consist(original_testcase, random_testcase2)
            fuzzer.make_inputs_and_weights_consist(original_testcase, simplified_testcase)
            fuzzer.make_inputs_and_weights_consist(original_testcase, complex_testcase)

            random_testcase1 = make_test_case_again(random_testcase1)
            random_testcase2 = make_test_case_again(random_testcase2)
            simplified_testcase = make_test_case_again(simplified_testcase)
            complex_testcase = make_test_case_again(complex_testcase)
        except BaseException as e:
            print(f"fail to make inputs and weights consistent: {e}")
            iteration -= 1
            continue

        try:
            fuzzer.make_output_consist(original_testcase, random_testcase1, graph_output_map1)
            fuzzer.make_output_consist(original_testcase, random_testcase2, graph_output_map2)
            fuzzer.make_output_consist(original_testcase, simplified_testcase, graph_output_map3)
            fuzzer.make_output_consist(original_testcase, complex_testcase, graph_output_map4)
        except AssertionError as e:
            # print(f"fail to make output consistent: {e}")
            # iteration -= 1
            # continue
            pass

        random1_log = process_log(run(fuzzer, random_testcase1, thread_id))
        random2_log = process_log(run(fuzzer, random_testcase2, thread_id))
        simplified_log = process_log(run(fuzzer, simplified_testcase, thread_id))
        complex_log = process_log(run(fuzzer, complex_testcase, thread_id))

        has_error = False
        for log in [random1_log, random2_log, complex_log, simplified_log]:
            if len(log) == 0:
                has_error = True
        if has_error:
            # I don't know why use safe_crash mode will cause a timeout
            print(f"timeout error or exception error")
            iteration -= 1
            continue

        edit_res_1 = edit_distance(random1_log, random2_log)
        edit_res_2 = edit_distance(simplified_log, complex_log)

        lcs_res_1 = lcs_distance(random1_log, random2_log)
        lcs_res_2 = lcs_distance(simplified_log, complex_log)

        if edit_res_1 != 0 and edit_res_2 != 0:
            print((edit_res_2 - edit_res_1) / edit_res_1, ((lcs_res_2 - lcs_res_1) / lcs_res_1))

        res_record.append((edit_res_1, edit_res_2, lcs_res_1, lcs_res_2, len(random1_log), len(random2_log),
                           len(simplified_log), len(complex_log)))

    return res_record


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    print(cfg)

    os.environ["ABLATION_EXP_RUN"] = "true"

    total_res_record = []
    max_works = MAX_WORKERS

    with ProcessPoolExecutor(max_workers=max_works) as executor:
        futures = []
        for i in range(max_works):
            futures.append(executor.submit(run_and_get_diff_res, cfg, MAX_ITERATION // max_works, i))

        for future in futures:
            total_res_record.extend(future.result())

    df = pd.DataFrame(total_res_record, columns=["edit_res_1", "edit_res_2", "lcs_res_1", "lcs_res_2", "len_random1",
                                                 "len_random2", "len_simplified", "len_complex"])

    output_dir_path = os.environ.get("OUTPUT_DIR")
    if output_dir_path is None:
        df.to_csv(f"ablation_exp_{MAX_ITERATION}.csv", index=False)
    else:
        dump_path = os.path.join(output_dir_path, "ablation_exp.csv")
        df.to_csv(dump_path, index=False)


if __name__ == '__main__':
    main()
