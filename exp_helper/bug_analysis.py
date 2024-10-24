import os
import sys
import argparse

from nnsmith.gir import GraphIR, InstIR
from nnsmith.materialize import Render, BugReport, TestCase, Model
from nnsmith.backends import BackendFactory
from equality_saturation_helper.minimizer import OperationLevel
from typing import List, Set, Tuple
import re
from tqdm import tqdm

ROOT_LEVEL = "origin"
CHAIN_END = ["core.Input", "core.Constant"]
RE_VAR = r"at (v\d+_\d+), (v\d+_\d+)"
RE_VAR2 = r"at (v\d+_\d+)"
RE_PATH = r"fuzz_report_(nnsmith|eq_fuzz|polyjuice|mtdlcomp)_(xla|tvm)_[01]"

# FALSE_POSITIVE = set(["tensorflow.Cholesky", "tensorflow.Eigh"])


FALSE_POSITIVE = set()


def find_inst_by_var(var: str, gir: GraphIR) -> InstIR:
    inst_id = int(var[1:].split("_")[0])
    return gir.find_inst_by_id(inst_id)


def find_mismatch_var(log) -> (str, str):
    matches = re.findall(RE_VAR, log)
    assert len(matches[0]) > 0
    mismatch_var_0 = matches[0][0]
    mismatch_var_1 = matches[0][1] if len(matches[0]) == 2 else None
    return mismatch_var_0, mismatch_var_1


def find_mismatch_var2(log) -> (str, str):
    matches = re.findall(RE_VAR2, log)
    assert len(matches[0]) > 0
    mismatch_var_0 = matches[0]
    return mismatch_var_0, None


def extract_op_chain(gir, mismatch_var_0) -> List[str]:
    op_chain = [ROOT_LEVEL]
    graphIR_0 = gir
    mismatch_inst_0 = find_inst_by_var(mismatch_var_0, graphIR_0)
    inst_stack = [mismatch_inst_0]
    while inst_stack:
        inst = inst_stack.pop()
        inst_name = inst.iexpr.op.name()
        if inst_name in CHAIN_END:
            continue
        op_chain.append(inst_name)
        for new_var in inst.iexpr.args:
            inst_stack.append(find_inst_by_var(new_var, graphIR_0))
    return op_chain


def analyze_crash(path) -> Set:
    crash_set = set()

    for root, dirs, files in os.walk(path):
        # print(len(crash_set))
        for directory in dirs:
            if "Symptom.EXCEPTION-Stage.EXECUTION" in directory:
                err_log_path = os.path.join(root, directory, "err.log")
                try:
                    with open(err_log_path, "r") as f:
                        crash_set.add(f.readlines()[-1][:60])
                except BaseException as e:
                    pass
            elif "Symptom.INCONSISTENCY-Stage.VERIFICATION" in directory:
                pass
            elif "Symptom.EQ_INCONSISTENCY-Stage.VERIFICATION" in directory:
                pass
            elif "Symptom.DOUBLE_INCONSISTENCY-Stage.VERIFICATION" in directory:
                pass
    # print(len(crash_set))
    # print("\n".join(crash_set))
    return crash_set


def chain_diff(chain1: Tuple[str], chain2: Tuple[str]) -> List[str]:
    diff_chain = []
    chain1 = list(chain1)
    for op in chain2:
        if op not in chain1:
            diff_chain.append(op)
        else:
            for i in range(len(chain1)):
                if chain1[i] == op:
                    chain1.pop(i)
                    break
    return diff_chain


def analyze_silent(path):
    path_split = os.path.split(path)
    folder_name = path_split[-1] if path_split[-1] != "" else path_split[-2]

    re_res = re.findall(RE_PATH, folder_name)
    # print(re_res)
    fuzzer = re_res[0][0]
    target = re_res[0][1]

    if fuzzer == "mtdlcomp":
        bug_path = os.path.join(path, target)
        if os.path.exists(bug_path):
            assert len(os.listdir(bug_path)) == 0
        return set(), set()

    if fuzzer == "polyjuice":
        fuzzer = "eq_fuzz"
    model_type = "onnx" if target == "tvm" else "tensorflow"
    backend_target = target
    model_type = Model.init(model_type, backend_target)
    factory = BackendFactory.init(
        backend_target,
        ad=None,
        target="cpu",
        optmax=True,
        parse_name=True,
    )
    # print(path)

    root_level = OperationLevel(ROOT_LEVEL, 0)

    diff_test_op_set = set()
    diff_test_record_chain = []
    eq_op_set = set()
    chain_map = {}
    eq_record_chain = []
    for root, dirs, files in tqdm(os.walk(path)):
        for directory in dirs:
            dir_path = os.path.join(root, directory)
            if "Symptom.INCONSISTENCY-Stage.VERIFICATION" in directory:
                try:
                    bug_report: BugReport = BugReport.load(model_type, dir_path)
                except BaseException as e:
                    continue
                gir = bug_report.testcase.model.gir
                if fuzzer == "eq_fuzz":
                    mismatch_vars = find_mismatch_var(bug_report.log)
                elif fuzzer == "nnsmith":
                    mismatch_vars = find_mismatch_var2(bug_report.log)
                else:
                    raise AssertionError
                op_chain = extract_op_chain(gir, mismatch_vars[0])
                flag = True
                for i in range(0, len(op_chain)):
                    if op_chain[i] in FALSE_POSITIVE:
                        flag = False
                        break
                    for j in range(i + 1, len(op_chain)):
                        op_check = (op_chain[i], op_chain[j])
                        if op_check in diff_test_op_set:
                            flag = False
                            break
                if flag:
                    diff_test_record_chain.append(op_chain)
                    # print(dir_path)
                    for i in range(0, len(op_chain)):
                        for j in range(i + 1, len(op_chain)):
                            op_check = (op_chain[i], op_chain[j])
                            diff_test_op_set.add(op_check)
            elif "Symptom.EQ_INCONSISTENCY-Stage.VERIFICATION" in directory or "Symptom.DOUBLE_INCONSISTENCY-Stage.VERIFICATION" in directory:
                try:
                    bug_report: BugReport = BugReport.load(model_type, dir_path)
                except BaseException as e:
                    # print(f"fail to load bug report: {e}\nskip it")
                    continue
                gir1 = bug_report.testcase.model.gir
                gir2 = bug_report.eq_testcase.model.gir
                mismatch_vars = find_mismatch_var(bug_report.log)
                op_chain1 = tuple(extract_op_chain(gir1, mismatch_vars[0]))
                op_chain2 = tuple(extract_op_chain(gir2, mismatch_vars[1]))
                diff_chain = tuple(chain_diff(op_chain1, op_chain2))
                flag = True
                fp_flag = False
                diff_chain_in_map = False
                if op_chain1 == op_chain2 or len(diff_chain) == 0:
                    continue

                if diff_chain in chain_map:
                    diff_chain_in_map = True
                    for i in range(0, len(op_chain1)):
                        if op_chain1[i] in FALSE_POSITIVE:
                            flag = False
                            fp_flag = True
                            break
                        for j in range(i + 1, len(op_chain1)):
                            op_check = (op_chain1[i], op_chain1[j])
                            if op_check in eq_op_set:
                                flag = False
                                break
                else:
                    diff_chain_in_map = False
                    chain_map[diff_chain] = set()
                if fp_flag:
                    continue

                assert diff_chain in chain_map
                if flag:
                    for i in range(0, len(op_chain1)):
                        for j in range(i + 1, len(op_chain1)):
                            op_check = (op_chain1[i], op_chain1[j])
                            eq_op_set.add(op_check)
                    chain_map[diff_chain].add(op_chain1)
                    # print(dir_path)
                else:
                    if not diff_chain_in_map:
                        chain_map[diff_chain].add(op_chain1)
                        # print(dir_path)

    # print(len(diff_test_record_chain))
    # print("\n".join([",".join(op_chain) for op_chain in diff_test_record_chain]))

    eq_record_chain = []
    for key, val in chain_map.items():
        for diff in val:
            eq_record_chain.append((key, diff))
    # print(len(eq_record_chain))
    # print(
    #     "\n".join(["(" + ",".join(op_chain[0]) + ") (" + ",".join(op_chain[1]) + ")" for op_chain in eq_record_chain]))

    tmp_diff_test_record_chain = []
    for chain in diff_test_record_chain:
        tmp_diff_test_record_chain.append(tuple(chain))
    diff_test_record_chain = tmp_diff_test_record_chain

    tmp_eq_record_chain = []
    for chain in eq_record_chain:
        tmp_eq_record_chain.append((tuple(chain[0]), tuple(chain[1])))
    eq_record_chain = tmp_eq_record_chain

    return set(diff_test_record_chain), set(eq_record_chain)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, help="bug type: crash or silent", required=True)
    parser.add_argument("--path", type=str, help="path to the bug report", required=True)
    args = parser.parse_args()
    if args.type == "crash":
        analyze_crash(args.path)
    elif args.type == "silent":
        analyze_silent(args.path)
    else:
        raise AssertionError("Invalid bug type: " + args.type + ", should be either crash or silent")


if __name__ == '__main__':
    main()
