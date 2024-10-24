from nnsmith.materialize import Render, BugReport, TestCase, Model
from nnsmith.materialize.onnx import ONNXModel
from nnsmith.materialize.torch import TorchModel
from nnsmith.backends import BackendFactory
from nnsmith.gir import GraphIR, InstIR
from equality_saturation_helper import bug_code_generation
import argparse
import os
from typing import Dict, List, Optional, Type
import numpy as np
import pickle
import traceback
import re
import queue
import copy

TAB = " " * 4
RE_MISMATCH = r"Mismatched elements: \d+ / \d+ \((.*?)%\)"
RE_VAR = r"at (v\d+_\d+), (v\d+_\d+)"
MISMATCH_SIG = "triggers assertion"
CHAIN_END = ["core.Input", "core.Constant"]
ROOT_LEVEL = "origin"
MISMATCH_THRESHOLD = 1


class OperationLevel():
    def __init__(self, op_name: str, level: int) -> None:
        self.op_name = op_name
        self.level = level
        self.bug_list: List[str] = []
        self.bug_feats = []
        self.sub_levels: List[OperationLevel] = []

    @property
    def count(self) -> int:
        return len(self.bug_list)

    def add(self, op_chain: List[str], features, report_path: str):
        assert op_chain and op_chain[0] == self.op_name
        if len(op_chain) == 1:
            self.bug_list.append(report_path)
            self.bug_feats.append(features)
        else:
            op_chain = op_chain[1:]
            next_op_name = op_chain[0]
            for sub_level in self.sub_levels:
                if next_op_name == sub_level.op_name:
                    sub_level.add(op_chain, features, report_path)
                    return
            new_level = OperationLevel(next_op_name, self.level + 1)
            new_level.add(op_chain, features, report_path)
            self.sub_levels.append(new_level)

    def pretty(self) -> str:
        output_str = ""
        if self.level == 0:
            for next_level in self.sub_levels:
                output_str += next_level.pretty()
        else:
            occupy = "  " * (self.level - 1)
            output_str += f"{occupy}- Level {self.level}(OP: {self.op_name}, count: {self.count})\n"
            for report, feats in zip(self.bug_list, self.bug_feats):
                feature_str = ""
                for idx, feat in enumerate(feats):
                    feature_str += str(idx)
                    feature_str += "*" if feat else "-"
                output_str += f"{occupy}  {feature_str},{report}\n"
            output_str += "\n"
            for next_level in self.sub_levels:
                output_str += next_level.pretty()
        return output_str


class Minimizer():
    def __init__(self,
                 model_type: Type["Model"] = None,
                 factory: BackendFactory = None,
                 input_path: str = None,
                 desired_code: str = None,
                 backend_target: str = None) -> None:
        self.model_type: Type["Model"] = model_type
        self.factory: BackendFactory = factory
        self.output_path: str = input_path
        os.chdir(self.output_path)
        self.desired_code: str = desired_code
        self.backend_target: str = backend_target
        self.bug_report: BugReport = BugReport.load(model_type, input_path)
        matches = re.findall(RE_VAR, self.bug_report.log)
        self.mismatch_var_0 = matches[0][0]
        self.mismatch_var_1 = matches[0][1]
        percentage_match = re.search(RE_MISMATCH, self.bug_report.log)
        if percentage_match:
            self.mismatch_percentage = float(percentage_match.group(1))
        else:
            self.mismatch_percentage = None
        self.mismatch_op = ""
        self.has_variable = False
        self.has_transpose = False

    def multiple_output(self) -> bool:
        return len(self.bug_report.other_info["output_map"]) > 1

    def find_inst_by_var(self, var: str, gir: GraphIR) -> InstIR:
        inst_id = int(var[1:].split("_")[0])
        return gir.find_inst_by_id(inst_id)

    def dump_model(self, root_folder: str, model: Model, input_dict: Type[Dict] = {}):
        if model and model.dotstring:
            model.dump_viz(os.path.join(root_folder, "graph.png"))
        model.dump(os.path.join(root_folder, model.name_prefix() + model.name_suffix()))
        with open(os.path.join(root_folder, "oracle.pkl"), "wb") as f:
            to_dump = {
                "input": input_dict,
                "output": None,
                "provider": None,
            }
            pickle.dump(to_dump, f)

    def generate_pyfile(self, model_0, model_1, output_map, file_fix, input_dict_0: Type[Dict] = {},
                        input_dict_1: Type[Dict] = {}):
        match self.desired_code:
            case "pt2":
                output_str = bug_code_generation.generation_torch_to_torch_compile(model_0, model_1, input_dict_0,
                                                                                   input_dict_1, output_map=output_map,
                                                                                   backend_target=self.backend_target)
            case "hidet":
                output_str = bug_code_generation.generation_torch_to_hidet(model_0, model_1, input_dict_0, input_dict_1,
                                                                           output_map=output_map,
                                                                           backend_target=self.backend_target)
            case "tensorrt":
                output_str = bug_code_generation.generation_onnx_to_tensorrt(model_0, model_1, input_dict_0,
                                                                             input_dict_1, output_map=output_map,
                                                                             backend_target=self.backend_target)
            case "infinitensor":
                output_str = bug_code_generation.generation_onnx_to_infinitensor(model_0, model_1, input_dict_0,
                                                                                 input_dict_1, output_map=output_map,
                                                                                 backend_target=self.backend_target)
            case "xla":
                if file_fix == "":
                    testcase0_path = os.path.join(self.output_path, f"t1")
                    testcase1_path = os.path.join(self.output_path, f"t2")
                else:
                    testcase0_path = os.path.join(self.output_path, f"{file_fix}0")
                    testcase1_path = os.path.join(self.output_path, f"{file_fix}1")
                    if not os.path.exists(testcase0_path):
                        os.makedirs(testcase0_path)
                    if not os.path.exists(testcase1_path):
                        os.makedirs(testcase1_path)
                    model_0.attach_viz(model_0.gir)
                    model_1.attach_viz(model_1.gir)
                    self.dump_model(testcase0_path, model_0, input_dict_0)
                    self.dump_model(testcase1_path, model_1, input_dict_1)
                output_str = bug_code_generation.generation_tensorflow_to_xla(
                    os.path.join(testcase0_path, "model", "tfnet"), os.path.join(testcase1_path, "model", "tfnet"),
                    input_dict_0, input_dict_1, output_map=output_map, backend_target=self.backend_target)
            case _:
                raise NotImplementedError

        with open(os.path.join(self.output_path, f"test{file_fix}.py"), "w+") as f:
            f.write(output_str)

        if "torch.nn.Parameter" in output_str:
            self.has_variable = True
        if "transpose" in output_str:
            self.has_transpose = True
        else:
            self.has_transpose = False

    @staticmethod
    def _assert_output_all_close(output_1: Dict[str, np.ndarray],
                                 output_2: Dict[str, np.ndarray],
                                 graph_output_map: Dict[str, str],
                                 equal_nan=True,
                                 rtol=1.0):
        for output_tensor_name_1, output_val_1 in output_1.items():
            if output_tensor_name_1 not in graph_output_map:
                continue
            output_tensor_name_2 = graph_output_map[output_tensor_name_1]
            if output_tensor_name_2 not in output_2:
                # this tensor is an intermediate node of the dst graph, but is an output node of src graph
                continue
            output_val_2 = output_2[output_tensor_name_2]
            np.testing.assert_allclose(
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

    def cmp_model_for_diff(self, model_0: Model, model_1: Model, input_dict_0: Dict[str, np.ndarray],
                           input_dict_1: Dict[str, np.ndarray], graph_output_map: Dict[str, str]) -> bool:
        try:
            model_1.make_weights_consist_with_other(model_0)
            testcase_0 = TestCase(model_0, model_0.make_oracle(input_dict_0))
            testcase_1 = TestCase(model_1, model_1.make_oracle(input_dict_1))
            try:
                self.make_output_consist(testcase_0, testcase_1, graph_output_map)
            except AssertionError as e:
                # print(e)
                print(traceback.format_exc())
                self.generate_pyfile(model_0, model_1, graph_output_map, "Asrt", input_dict_0, input_dict_1)
                print("Assertion on ", self.output_path)
                return False
            except Exception as e:
                print(e)
                self.generate_pyfile(model_0, model_1, graph_output_map, "Err", input_dict_0, input_dict_1)
                return False
            res0 = self.factory.checked_compile_and_exec(testcase_0, crash_safe=False)
            res1 = self.factory.checked_compile_and_exec(testcase_1, crash_safe=False)
            if isinstance(res0, BugReport):
                return False
            if isinstance(res1, BugReport):
                return False
        except Exception as e:
            return False

        try:
            self._assert_output_all_close(res0, res1, graph_output_map)
        except:
            return True
        return False

    def update_model(self, model_new_0=None, model_new_1=None, output_map_new=None):
        if model_new_0:
            self.bug_report.testcase.model = model_new_0
        if model_new_1:
            self.bug_report.eq_testcase.model = model_new_1
        if output_map_new:
            self.bug_report.other_info["output_map"] = output_map_new

    def cut_leaf_chain(self, gir: GraphIR, target_inst: InstIR, input_dict):
        insts_queue = queue.Queue(gir.n_inst())
        if target_inst.no_users():
            insts_queue.put(target_inst)
            exclu_inst = gir.leaf_inst()
            exclu_inst.remove(target_inst)

        while not insts_queue.empty():
            inst_to_cut = insts_queue.get()
            if inst_to_cut.iexpr.op.name == "core.Input":
                del input_dict[f"v{inst.identifier}_0"]
            gir.remove_unused(inst_to_cut)
            if insts_queue.empty():
                for inst in gir.leaf_inst():
                    if inst not in exclu_inst:
                        insts_queue.put(inst)
        return gir

    def eliminate_uncmped_output(self) -> bool:
        ret = False
        output_map = self.bug_report.other_info["output_map"]
        output_var_0 = self.bug_report.testcase.model.gir.leaf_var()
        output_var_1 = self.bug_report.eq_testcase.model.gir.leaf_var()
        for k, v in output_map.items():
            if k in output_var_0 and v in output_var_1:
                output_var_0.remove(k)
                output_var_1.remove(v)

        model_1 = self.bug_report.eq_testcase.model
        for var in output_var_0:
            graphIR_0 = copy.deepcopy(self.bug_report.testcase.model.gir)
            input_dict_0 = copy.deepcopy(self.bug_report.testcase.oracle.input)
            input_dict_1 = copy.deepcopy(self.bug_report.eq_testcase.oracle.input)

            self.cut_leaf_chain(graphIR_0, self.find_inst_by_var(var, graphIR_0), input_dict_0)
            model_new_0 = self.model_type.from_gir(graphIR_0)
            res_cmp = self.cmp_model_for_diff(model_new_0, model_1, input_dict_0, input_dict_1, output_map)
            if res_cmp:
                self.update_model(model_new_0=model_new_0)
                self.bug_report.testcase.oracle.input = input_dict_0
                ret = True

        model_0 = self.bug_report.testcase.model
        for var in output_var_1:
            graphIR_1 = copy.deepcopy(self.bug_report.eq_testcase.model.gir)
            input_dict_0 = copy.deepcopy(self.bug_report.testcase.oracle.input)
            input_dict_1 = copy.deepcopy(self.bug_report.eq_testcase.oracle.input)

            self.cut_leaf_chain(graphIR_1, self.find_inst_by_var(var, graphIR_1), input_dict_1)
            model_new_1 = self.model_type.from_gir(graphIR_1)
            res_cmp = self.cmp_model_for_diff(model_0, model_new_1, input_dict_0, input_dict_1, output_map)
            if res_cmp:
                self.update_model(model_new_1=model_new_1)
                self.bug_report.eq_testcase.oracle.input = input_dict_1
                ret = True
        return ret

    def eliminate_useless_output(self) -> bool:
        ret = False
        if self.multiple_output():
            output_map_origin = copy.deepcopy(self.bug_report.other_info["output_map"])
            del output_map_origin[self.mismatch_var_0]

            for key, val in output_map_origin.items():
                output_map_new = copy.deepcopy(self.bug_report.other_info["output_map"])
                del output_map_new[key]
                input_dict_0 = copy.deepcopy(self.bug_report.testcase.oracle.input)
                input_dict_1 = copy.deepcopy(self.bug_report.eq_testcase.oracle.input)
                graphIR_0 = copy.deepcopy(self.bug_report.testcase.model.gir)
                graphIR_1 = copy.deepcopy(self.bug_report.eq_testcase.model.gir)
                try:
                    gir_tmp_0 = self.cut_leaf_chain(graphIR_0, self.find_inst_by_var(key, graphIR_0), input_dict_0)
                    gir_tmp_1 = self.cut_leaf_chain(graphIR_1, self.find_inst_by_var(val, graphIR_1), input_dict_1)
                except:
                    print("output_map error!")
                    if key in self.bug_report.other_info["output_map"]:
                        del self.bug_report.other_info["output_map"][key]
                    continue
                model_new_0 = self.model_type.from_gir(gir_tmp_0)
                model_new_1 = self.model_type.from_gir(gir_tmp_1)
                res_cmp = self.cmp_model_for_diff(model_new_0, model_new_1, input_dict_0, input_dict_1, output_map_new)

                if res_cmp:
                    self.update_model(model_new_0, model_new_1, output_map_new)
                    self.bug_report.testcase.oracle.input = input_dict_0
                    self.bug_report.eq_testcase.oracle.input = input_dict_1
                    ret = True
        return ret

    def trace_root_op(self, gir_0: GraphIR, gir_1: GraphIR) -> bool:
        ret = False
        graphIR_0 = copy.deepcopy(gir_0)
        graphIR_1 = copy.deepcopy(gir_1)
        mismatch_inst_0 = self.find_inst_by_var(self.mismatch_var_0, graphIR_0)
        mismatch_inst_1 = self.find_inst_by_var(self.mismatch_var_1, graphIR_1)
        if not mismatch_inst_0 or not mismatch_inst_1:
            return ret
        self.mismatch_op = mismatch_inst_0.iexpr.op

        if mismatch_inst_0.no_users() and mismatch_inst_1.no_users():
            if len(mismatch_inst_0.iexpr.args) == 1 and len(mismatch_inst_1.iexpr.args) == 1:
                var_new_0 = mismatch_inst_0.iexpr.args[0]
                graphIR_0.remove_unused(mismatch_inst_0)
                var_new_1 = mismatch_inst_1.iexpr.args[0]
                graphIR_1.remove_unused(mismatch_inst_1)

                output_map_new = copy.deepcopy(self.bug_report.other_info["output_map"])
                del output_map_new[self.mismatch_var_0]
                if var_new_0 in graphIR_0.leaf_var() and var_new_1 in graphIR_1.leaf_var():
                    output_map_new[var_new_0] = var_new_1

                model_new_0 = self.model_type.from_gir(graphIR_0)
                model_new_1 = self.model_type.from_gir(graphIR_1)
                input_dict_0 = copy.deepcopy(self.bug_report.testcase.oracle.input)
                input_dict_1 = copy.deepcopy(self.bug_report.eq_testcase.oracle.input)

                res_cmp = self.cmp_model_for_diff(model_new_0, model_new_1, input_dict_0, input_dict_1, output_map_new)

                if res_cmp:
                    self.update_model(model_new_0, model_new_1, output_map_new)
                    self.mismatch_var_0 = var_new_0
                    self.mismatch_var_1 = var_new_1
                    self.trace_root_op(self.bug_report.testcase.model.gir, self.bug_report.eq_testcase.model.gir)
                    ret = True
        return ret

    def extract_op_chain(self) -> List[str]:
        op_chain = [ROOT_LEVEL]
        graphIR_0 = self.bug_report.testcase.model.gir
        mismatch_inst_0 = self.find_inst_by_var(self.mismatch_var_0, graphIR_0)
        inst_stack = [mismatch_inst_0]
        while inst_stack:
            inst = inst_stack.pop()
            inst_name = inst.iexpr.op.name()
            if inst_name in CHAIN_END:
                continue
            op_chain.append(inst_name)
            for new_var in inst.iexpr.args:
                inst_stack.append(self.find_inst_by_var(new_var, graphIR_0))
        return op_chain

    def run(self) -> List[str]:
        self.generate_pyfile(self.bug_report.testcase.model, self.bug_report.eq_testcase.model,
                             self.bug_report.other_info["output_map"], "", self.bug_report.testcase.oracle.input,
                             self.bug_report.eq_testcase.oracle.input)
        minimized = False
        try:
            minimized = self.eliminate_uncmped_output() or minimized
            minimized = self.eliminate_useless_output() or minimized
            minimized = self.trace_root_op(self.bug_report.testcase.model.gir,
                                           self.bug_report.eq_testcase.model.gir) or minimized
        except Exception as e:
            print(e)
            print(f"Minimization error in {self.output_path}")

        try:
            op_chain = self.extract_op_chain()
        except Exception as e:
            op_chain = [ROOT_LEVEL, "ERROR"]
            print(e)
            print(f"Opchain extraction error in {self.output_path}")
            # exit()
        features = [self.has_transpose, (self.mismatch_percentage == None),
                    (self.mismatch_percentage and self.mismatch_percentage >= 50)]
        if minimized:
            features.append(True)
            self.generate_pyfile(self.bug_report.testcase.model, self.bug_report.eq_testcase.model,
                                 self.bug_report.other_info["output_map"], "Min", self.bug_report.testcase.oracle.input,
                                 self.bug_report.eq_testcase.oracle.input)
        else:
            features.append(False)
        return op_chain, features


def main():
    parser = argparse.ArgumentParser(description="minimize the graph for de-duplication")
    parser.add_argument("-i", "--input_path", type=str, help="directory path of bug report", required=True)
    parser.add_argument("-m", "--model_type", type=str, help="the model type of the bug report", required=True)
    parser.add_argument("-d", "--desired_code", type=str, help="the code you want to output (tvm/torch/torchcompile)",
                        required=True)
    parser.add_argument("-b", "--backend_target", type=str, help="backend target (cpu/cuda)", required=True)
    parser.add_argument("-o", "--output_path", type=str, help="output directory for code", required=True)
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    model_type = args.model_type
    backend_target = args.backend_target
    desired_code = args.desired_code
    model_init = Model.init(model_type, backend_target)
    factory = BackendFactory.init(
        desired_code,
        ad=None,
        target=backend_target,
        optmax=True,
        parse_name=True,
    )

    root_level = OperationLevel(ROOT_LEVEL, 0)
    for root, dirs, files in os.walk(input_path):
        for dir in dirs:
            if "bug-Symptom.EQ_INCONSISTENCY-Stage" in dir:
                report_path = os.path.join(root, dir)
                print(f"========processing {report_path}  ========")
                try:
                    m = Minimizer(model_init, factory, report_path, desired_code, backend_target)
                except:
                    print(f"LOADING ERROR: {report_path}")
                    continue
                if m.mismatch_percentage and m.mismatch_percentage < MISMATCH_THRESHOLD:
                    continue
                op_chain, feats = m.run()
                root_level.add(op_chain, feats, report_path)

    with open(output_path, "w+") as f:
        f.write(root_level.pretty())


if __name__ == '__main__':
    main()
