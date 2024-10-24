from nnsmith.cli.equivalent_fuzz import EquivalentFuzzingLoop
from nnsmith.materialize import TestCase
from nnsmith.logging import FUZZ_LOG
from omegaconf import OmegaConf
import time
from equality_saturation_helper.helper import FFIHelper, EquivalentGraphHelper
import random
from enum import Enum


class RunType(Enum):
    OK = 1,
    MAKE_TESTCASE_FAIL = 2,
    RUN_TESTCASE_FAIL = 3,


class TestCorrectnessOfEQHelper(EquivalentFuzzingLoop):
    def run(self):
        print(self.opset)
        start_time = time.time()
        ffi_helper = FFIHelper()
        assert ffi_helper.is_helper_loaded()
        iteration = 0
        passed = 0
        bug = 0
        while time.time() - start_time < self.timeout_s:
            print(f"pass/bug/iteration: {passed}/{bug}/{iteration}")
            iteration += 1

            pass_flag = 0
            bug_flag = 0

            seed = random.getrandbits(32)
            ir = self.make_gir(seed=seed)
            eq_graph_helper = EquivalentGraphHelper(ffi_helper=ffi_helper, gir=ir)
            eq_graph_helper.initialize_inner_graph()
            eq_graph_helper.run_saturation()
            eq_graph_helper.test_helper_lib()

            ir2, graph_output_map = eq_graph_helper.randomly_generate_an_equivalent_graph()
            run_res = self.cmp_result(ir, ir2, graph_output_map)
            if run_res == RunType.OK or run_res == RunType.MAKE_TESTCASE_FAIL:
                pass_flag += 1
            elif run_res == RunType.RUN_TESTCASE_FAIL:
                bug_flag += 1
                FUZZ_LOG.warning(f"Failed model seed: {seed}")

            ir3, graph_output_map = eq_graph_helper.find_the_most_complex_equivalent_graph()
            run_res = self.cmp_result(ir, ir3, graph_output_map)
            if run_res == RunType.OK or run_res == RunType.MAKE_TESTCASE_FAIL:
                pass_flag += 1
            elif run_res == RunType.RUN_TESTCASE_FAIL:
                bug_flag += 1
                FUZZ_LOG.warning(f"Failed model seed: {seed}")

            ir4, graph_output_map = eq_graph_helper.find_the_most_simplified_equivalent_graph()
            run_res = self.cmp_result(ir, ir4, graph_output_map)
            if run_res == RunType.OK or run_res == RunType.MAKE_TESTCASE_FAIL:
                pass_flag += 1
            elif run_res == RunType.RUN_TESTCASE_FAIL:
                bug_flag += 1
                FUZZ_LOG.warning(f"Failed model seed: {seed}")

            if pass_flag == 3:
                passed += 1

            if bug_flag > 0:
                bug += 1

    def cmp_result(self, ir1, ir2, graph_output_map) -> RunType:
        try:
            testcase_1 = self.gir_to_testcase(ir1)
            testcase_2 = self.gir_to_testcase(ir2)
        except BaseException as e:
            print(e)
            return RunType.MAKE_TESTCASE_FAIL
        self.make_inputs_and_weights_consist(testcase_1, testcase_2)
        testcase_1 = TestCase(testcase_1.model, testcase_1.model.make_oracle(testcase_1.oracle.input))
        testcase_2 = TestCase(testcase_2.model, testcase_2.model.make_oracle(testcase_2.oracle.input))
        try:
            self.make_output_consist(testcase_1, testcase_2, graph_output_map)
        except BaseException as e:
            print(testcase_1.model.gir.to_dot())
            print(testcase_2.model.gir.to_dot())
            return RunType.MAKE_TESTCASE_FAIL
        res = self.validate_results(testcase_1, testcase_2, graph_output_map)
        if res is not None:
            print(res)
            return RunType.RUN_TESTCASE_FAIL
        else:
            return RunType.OK

    def yield_manual_test(self):
        pass


def main():
    cfg = {'model': {'type': 'torch', 'path': '???'},
           'mgen': {'max_nodes': 10, 'timeout_ms': 10000, 'vulops': False, 'method': 'symbolic-cinit',
                    'save': 'nnsmith_output', 'seed': None, 'max_elem_per_tensor': 65536, 'rank_choices': [2, 4],
                    'dtype_choices': None, 'include': None, 'exclude': ["Round", "Cast"], 'patch_requires': [],
                    'grad_check': False},
           'backend': {'type': 'torchjit', 'optmax': True, 'target': 'cpu'}, 'ad': {'type': None},
           'cache': {'topset': True},
           'debug': {'viz': True, 'viz_fmt': 'png'},
           'fuzz': {'time': '12h', 'root': 'eq_test_output', 'seed': None, 'crash_safe': False, 'test_timeout': None,
                    'save_test': None}, 'filter': {'type': [], 'patch': []},
           'cmp': {'equal_nan': True, 'raw_input': None, 'oracle': None,
                   'with': {'type': None, 'optmax': True, 'target': 'cpu'}, 'seed': None, 'bug_presence': 'report',
                   'save': None}}
    cfg = OmegaConf.create(cfg)
    test = TestCorrectnessOfEQHelper(cfg)
    test.run()


if __name__ == "__main__":
    main()
