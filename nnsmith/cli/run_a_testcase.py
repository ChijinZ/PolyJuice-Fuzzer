import sys
import os

from equivalent_fuzz import EquivalentFuzzingLoop, TestCase, BugReport
import pickle


def main():
    fuzzer_object_path = sys.argv[1]
    test_case_path = sys.argv[2]

    tmp_file_path = None

    if len(sys.argv) == 4:
        tmp_file_path = sys.argv[3]

    assert os.path.exists(fuzzer_object_path), f"{fuzzer_object_path} does not exist"

    with open(fuzzer_object_path, "rb") as f:
        fuzzer_object: EquivalentFuzzingLoop = pickle.load(f)

    model_type = fuzzer_object.ModelType
    testcase = TestCase.load(model_type, test_case_path)

    if tmp_file_path is None:
        exit(101)

    res = fuzzer_object.execute_testcase(testcase, crash_safe=True, file_path_for_subprocess_log=tmp_file_path,
                                         timeout=20)
    if isinstance(res, BugReport):
        exit(100)


if __name__ == '__main__':
    main()
