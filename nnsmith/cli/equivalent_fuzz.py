from nnsmith.cli.fuzz import *
from nnsmith.cli.model_exec import check_result
from nnsmith.materialize import BugReport, Symptom, Stage
from nnsmith.gir import GraphIR
from typing import Optional, Union, Dict
import numpy as np
from equality_saturation_helper.helper import EquivalentGraphHelper, FFIHelper, UnreachableException
from numpy import testing
import re

RE_MISMATCH = r"Mismatched elements: \d+ / \d+ \((.*?)%\)"
VALIDATE_TIMES = 1


class EquivalentFuzzingLoop(FuzzingLoop):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        save_eq_is_wrong = os.getenv("SAVE_EQ_IS_WRONG")
        save_exception = os.getenv("SAVE_EXCEPTION")
        self.save_eq_is_wrong = False if save_eq_is_wrong is not None and save_eq_is_wrong == "false" else True
        self.save_exception = False if save_exception is not None and save_exception == "false" else True
        self.re_pattern = re.compile(RE_MISMATCH)

    def execute_testcase(self,
                         testcase: TestCase,
                         crash_safe: bool = False,
                         timeout: Optional[int] = None,
                         file_path_for_subprocess_log=None
                         ) -> Union[Dict[str, np.ndarray], BugReport]:
        bug_or_res = self.factory.checked_compile_and_exec(
            testcase, crash_safe=crash_safe, timeout=timeout, file_path_for_subprocess_log=file_path_for_subprocess_log
        )
        return bug_or_res

    def make_gir(self, seed) -> GraphIR:
        mgen_cfg = self.cfg["mgen"]
        gen = model_gen(
            opset=self.opset,
            method=mgen_cfg["method"],
            seed=seed,
            max_elem_per_tensor=mgen_cfg["max_elem_per_tensor"],
            max_nodes=mgen_cfg["max_nodes"],
            timeout_ms=mgen_cfg["timeout_ms"],
            rank_choices=mgen_cfg["rank_choices"],
            dtype_choices=mgen_cfg["dtype_choices"],
        )
        ir = gen.make_concrete()
        ir.wellform_repair()
        return ir

    @staticmethod
    def _assert_output_all_close(output_1: Dict[str, np.ndarray],
                                 output_2: Dict[str, np.ndarray],
                                 graph_output_map: Dict[str, str],
                                 equal_nan=True,
                                 rtol=0.01):
        for output_tensor_name_1, output_val_1 in output_1.items():
            output_tensor_name_2 = graph_output_map[output_tensor_name_1]
            if output_tensor_name_2 not in output_2:
                # this tensor is an intermediate node of the dst graph, but is an output node of src graph
                continue
            output_val_2 = output_2[output_tensor_name_2]
            testing.assert_allclose(
                output_val_1,
                output_val_2,
                equal_nan=equal_nan,
                rtol=rtol,
                err_msg=f"{output_val_1} != {output_val_2} at {output_tensor_name_1}, {output_tensor_name_2}",
            )

    def make_output_consist(self, src_testcase: TestCase, dst_testcase: TestCase, graph_output_map: Dict[str, str]):
        graph_input_map = src_testcase.model.corresponding_input_map(dst_testcase.model)
        self._assert_output_all_close(src_testcase.oracle.input, dst_testcase.oracle.input, graph_input_map,
                                      rtol=0.0001)

        self._assert_output_all_close(src_testcase.oracle.output,
                                      dst_testcase.oracle.output,
                                      graph_output_map)

        # if isinstance(src_testcase.model, ONNXModel):
        #     if isinstance(self.factory, TVM):
        #         another_backend = BackendFactory.init("onnxruntime")
        #     elif isinstance(self.factory, ORT):
        #         another_backend = BackendFactory.init("tvm")
        #     else:
        #         raise NotImplementedError
        #
        #     new_dst_testcase = another_backend.make_testcase(dst_testcase.model, dst_testcase.oracle.input)
        #     dst_testcase.oracle = new_dst_testcase.oracle
        #
        #     new_src_testcase = another_backend.make_testcase(src_testcase.model, src_testcase.oracle.input)
        #     src_testcase.oracle = new_src_testcase.oracle
        #
        # graph_input_map = src_testcase.model.corresponding_input_map(dst_testcase.model)
        # self._assert_output_all_close(src_testcase.oracle.input, dst_testcase.oracle.input, graph_input_map,
        #                               rtol=0.0001)
        # self._assert_output_all_close(src_testcase.oracle.output,
        #                               dst_testcase.oracle.output,
        #                               graph_output_map)

    @staticmethod
    def make_inputs_and_weights_consist(src_testcase: TestCase, dst_testcase: TestCase):
        src_model: Model = src_testcase.model
        dst_model: Model = dst_testcase.model

        # deal with weights
        dst_model.make_weights_consist_with_other(src_model)

        # deal with inputs
        name_map: Dict[str, str] = src_model.corresponding_input_map(dst_model)
        for src_input_name, src_ndarray in src_testcase.oracle.input.items():
            dst_input_name = name_map[src_input_name]
            assert dst_testcase.oracle.input[dst_input_name].shape == src_ndarray.shape
            dst_testcase.oracle.input[dst_input_name] = src_ndarray

    def gir_to_testcase(self, ir: GraphIR) -> TestCase:
        model = self.ModelType.from_gir(ir)
        if self.cfg["debug"]["viz"]:
            model.attach_viz(ir)

        # model.refine_weights()  # either random generated or gradient-based.
        model.set_grad_check(self.cfg["mgen"]["grad_check"])
        # print([(name, param) for name, param in model.torch_model.named_parameters()])
        oracle = model.make_oracle()
        return TestCase(model, oracle)

    def validate_results(self, testcase1: TestCase, testcase2: TestCase,
                         graph_output_map: Dict[str, str]) -> Optional[BugReport]:
        cmp_cfg = self.cfg["cmp"]

        graph_input_map = testcase1.model.corresponding_input_map(testcase2.model)
        self._assert_output_all_close(testcase1.oracle.input, testcase2.oracle.input, graph_input_map, rtol=0.0001)

        res1 = self.execute_testcase(testcase1)
        res2 = self.execute_testcase(testcase2)
        # if self.save_exception:
        #     if not check_result(res1, output_dir, filters=self.filters, msg="testcase1"):
        #         return False
        #     if not check_result(res2, output_dir, filters=self.filters, msg="testcase2"):
        #         return False
        # else:
        if isinstance(res1, BugReport):
            return res1
        if isinstance(res2, BugReport):
            return res2

        try:
            self._assert_output_all_close(res1, res2, graph_output_map)
        except AssertionError:
            try:
                self._assert_output_all_close(testcase1.oracle.output, testcase2.oracle.output, graph_output_map)
                bug_report = BugReport(
                    testcase=testcase1,
                    eq_testcase=testcase2,
                    system=self.factory.system_name,
                    symptom=Symptom.EQ_INCONSISTENCY,
                    stage=Stage.VERIFICATION,
                    log=traceback.format_exc(),
                    version=self.factory.version,
                    other_info={"output_map": graph_output_map}
                )
                return bug_report
                # check_result(bug_report,
                #              odir=output_dir,
                #              filters=self.filters,
                #              msg=f"Equivalent Check")
            except AssertionError:
                bug_report = BugReport(
                    testcase=testcase1,
                    eq_testcase=testcase2,
                    system=self.factory.system_name,
                    symptom=Symptom.EQ_IS_WRONG,
                    stage=Stage.VERIFICATION,
                    log=traceback.format_exc(),
                    version=self.factory.version,
                    other_info={"output_map": graph_output_map}
                )
                return bug_report
                # if self.save_eq_is_wrong:
                #     check_result(bug_report,
                #                  odir=output_dir,
                #                  filters=self.filters,
                #                  msg=f"Equivalent Double Check")
            except BaseException as e:
                raise e
        return None

    def run(self):
        start_time = time.time()
        ffi_helper = FFIHelper()
        assert ffi_helper.is_helper_loaded()
        passed = 0
        total = 0
        try:
            while time.time() - start_time < self.timeout_s:
                FUZZ_LOG.info(f"pass rate: {passed}/{total}")
                total += 1
                seed = random.getrandbits(32)
                FUZZ_LOG.debug(f"Making testcase with seed: {seed}")

                time_stat = {}

                gen_start = time.time()
                try:
                    need_make_ir = True
                    while need_make_ir:
                        try:
                            ir = self.make_gir(seed=seed)
                            need_make_ir = False
                        except BaseException as e:
                            seed = random.getrandbits(32)
                    eq_graph_helper = EquivalentGraphHelper(ffi_helper=ffi_helper, gir=ir)
                    # FUZZ_LOG.info(f"[{time.time() - start_time}]memory usage 1: {psutil.Process().memory_info().rss}")
                    eq_graph_helper.initialize_inner_graph()
                    saturation_start = time.time()
                    eq_graph_helper.run_saturation()
                    time_stat["saturation"] = time.time() - saturation_start
                    # eq_graph_helper.test_helper_lib()
                    generate_egraph_start = time.time()
                    # ir2, graph_output_map = eq_graph_helper.randomly_generate_an_equivalent_graph()
                    ir2, graph_output_map = eq_graph_helper.find_the_most_complex_equivalent_graph()
                    if ir2 is None:
                        continue
                    # FUZZ_LOG.info(f"[{time.time() - start_time}]memory usage 2: {psutil.Process().memory_info().rss}")
                    # del eq_graph_helper
                    # FUZZ_LOG.info(f"[{time.time() - start_time}]memory usage 3: {psutil.Process().memory_info().rss}")
                    time_stat["egraph_gen"] = time.time() - generate_egraph_start
                    testcase_1 = self.gir_to_testcase(ir)
                    testcase_2 = self.gir_to_testcase(ir2)

                    # make weights consistent
                    self.make_inputs_and_weights_consist(testcase_1, testcase_2)

                    # make input consistent
                    # testcase_1 = TestCase(testcase_1.model, testcase_1.model.make_oracle(testcase_1.oracle.input))
                    testcase_2 = TestCase(testcase_2.model, testcase_2.model.make_oracle(testcase_2.oracle.input))

                    # make output consistent
                    try:
                        self.make_output_consist(testcase_1, testcase_2, graph_output_map)
                    except AssertionError as e:
                        bug_report = BugReport(
                            testcase=testcase_1,
                            eq_testcase=testcase_2,
                            system=self.factory.system_name,
                            symptom=Symptom.EQ_IS_WRONG,
                            stage=Stage.VERIFICATION,
                            log=traceback.format_exc(),
                            version=self.factory.version,
                            other_info={"output_map": graph_output_map}
                        )
                        if self.save_eq_is_wrong:
                            check_result(bug_report,
                                         odir=self.status.get_next_bug_path(),
                                         filters=self.filters,
                                         msg=f"Equivalent Check for EQ_IS_WRONG")
                            self.status.n_bugs += 1
                        raise e
                except InternalError as e:
                    raise e  # propagate internal errors
                except Exception as e:
                    FUZZ_LOG.error(
                        f"`make_testcase` failed with seed {seed}."
                        f"Error during Generation: error msg: {e}"
                    )
                    FUZZ_LOG.error(traceback.format_exc())
                    self.status.n_fail_make_test += 1
                    continue
                time_stat["gen"] = time.time() - gen_start

                eval_start = time.time()

                best_bug_report: Optional[BugReport] = None
                best_miss_match_num: float = 0.0
                for i in range(VALIDATE_TIMES):
                    FUZZ_LOG.info(f"iteration: {i}")
                    res = self.validate_results(testcase_1, testcase_2, graph_output_map)
                    if isinstance(res, BugReport):
                        if res.symptom == Symptom.EQ_IS_WRONG or res.symptom == Symptom.EXCEPTION:
                            best_bug_report = res
                            break
                        elif res.symptom == Symptom.EQ_INCONSISTENCY:
                            percentage_match = self.re_pattern.search(res.log)
                            FUZZ_LOG.info(f"percentage_match: {percentage_match}")
                            if percentage_match is None:
                                # perhaps it is nan or shape mismatch, just record it
                                best_bug_report = res
                                break
                            else:
                                # mismatch, we should finetune the inputs
                                mismatch_num = float(percentage_match.group(1))
                                if mismatch_num > best_miss_match_num:
                                    best_miss_match_num = mismatch_num
                                    best_bug_report = res
                                FUZZ_LOG.info(best_miss_match_num)
                        else:
                            # there should not have other exception
                            raise UnreachableException()
                    else:
                        if best_miss_match_num > 0.0:
                            continue
                        else:
                            # not-a-bug, we don't need to fine-tune it
                            break

                    # we can assert that they will not trigger any crash
                    testcase_1 = TestCase(testcase_1.model, testcase_1.model.make_oracle())
                    testcase_2 = TestCase(testcase_2.model, testcase_2.model.make_oracle())
                    self.make_inputs_and_weights_consist(testcase_1, testcase_2)
                    testcase_2 = TestCase(testcase_2.model, testcase_2.model.make_oracle(testcase_2.oracle.input))

                    try:
                        self.make_output_consist(testcase_1, testcase_2, graph_output_map)
                    except AssertionError as e:
                        # a specific input can make eq wrong
                        best_bug_report = BugReport(
                            testcase=testcase_1,
                            eq_testcase=testcase_2,
                            system=self.factory.system_name,
                            symptom=Symptom.EQ_IS_WRONG,
                            stage=Stage.VERIFICATION,
                            log=traceback.format_exc(),
                            version=self.factory.version,
                            other_info={"output_map": graph_output_map}
                        )
                        break

                if isinstance(best_bug_report, BugReport):
                    self.status.n_bugs += 1
                    FUZZ_LOG.warning(f"Failed model seed: {seed}")
                    output_dir = self.status.get_next_bug_path()

                    if best_bug_report.symptom == Symptom.EXCEPTION and self.save_exception:
                        check_result(best_bug_report,
                                     odir=output_dir,
                                     filters=self.filters,
                                     msg=f"throw an exception during testing")
                    elif best_bug_report.symptom == Symptom.EQ_IS_WRONG and self.save_eq_is_wrong:
                        check_result(best_bug_report,
                                     odir=output_dir,
                                     filters=self.filters,
                                     msg=f"eq is wrong")
                    elif best_bug_report.symptom == Symptom.EQ_INCONSISTENCY:
                        check_result(best_bug_report,
                                     odir=output_dir,
                                     filters=self.filters,
                                     msg=f"eq graph have consistent outputs")

                time_stat["eval"] = time.time() - eval_start

                if self.save_test:
                    save_start = time.time()
                    testcase_dir = os.path.join(
                        self.save_test, f"{time.time() - start_time:.3f}"
                    )
                    mkdir(testcase_dir)
                    tmp, testcase_1.model.dotstring = testcase_1.model.dotstring, None
                    testcase_1.dump(testcase_dir)
                    testcase_1.model.dotstring = tmp
                    time_stat["save"] = time.time() - save_start

                time_stat_list = [(k, time_stat[k]) for k in ["gen", "saturation", "egraph_gen", "eval"]]
                FUZZ_LOG.info(
                    f"Timing: {''.join(f'{k}: {1000 * v:.1f}ms, ' for (k, v) in time_stat_list)}"
                )
                self.status.n_testcases += 1
                passed += 1
        except KeyboardInterrupt:
            pass
        FUZZ_LOG.info(f"Total {self.status.n_testcases} testcases executed.")
        FUZZ_LOG.info(f"Total {self.status.n_bugs} bugs found.")
        FUZZ_LOG.info(f"Total {self.status.n_fail_make_test} failed to make testcases.")


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    print(cfg)
    EquivalentFuzzingLoop(cfg).run()


if __name__ == "__main__":
    main()
