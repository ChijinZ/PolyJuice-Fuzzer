from cffi import FFI
import jsonpickle
from nnsmith.gir import GraphIR, InstIR, InstExpr
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.abstract.op import *
from nnsmith.logging import HELPER_LOG
from nnsmith.materialize import Oracle, Model
from typing import Optional, Dict, List, Tuple, Set


class UnreachableException(Exception):
    pass


class FFIHelper:
    def __init__(self):
        self.ffi = FFI()
        self.ffi.cdef("""
                    void init_helper();
                    void* initialize_graph(char* graph_json_str_);
                    void output_self_graph(void* src_graph, char* out_graph_json_str);
                    void run_saturation(void* graph_);
                    void deallocate_graph(void* graph_);
                    void randomly_find_an_equivalent_graph(void* src_graph, char* out_graph_json_str);
                    void find_the_most_simplified_equivalent_graph(void* src_graph, char* out_graph_json_str);
                    void find_the_most_complex_equivalent_graph(void* src_graph, char* out_graph_json_str);
                """)

        helper_path = os.getenv("EQ_HELPER_PATH")
        if helper_path:
            self.Clib = self.ffi.dlopen(helper_path)
            self.Clib.init_helper()
            HELPER_LOG.info(f"successfully load helper from {helper_path}")
        else:
            HELPER_LOG.error("did not set EQ_HELPER_PATH")

    def is_helper_loaded(self) -> bool:
        return self.Clib is not None


class EquivalentGraphHelper:
    def __init__(self, ffi_helper: FFIHelper, gir: GraphIR):
        self.ffi = ffi_helper.ffi
        self.Clib = ffi_helper.Clib
        self.gir: Optional[GraphIR] = gir
        self.inner_graph = None
        self.op_info: List[Union[AbsOpBase, Placeholder]] = []  # operator that may be used later

    def __del__(self):
        if self.Clib and self.inner_graph is not None:
            self.Clib.deallocate_graph(self.inner_graph)

    def initialize_inner_graph(self):
        gir_json = self.trans_gir_to_json(self.gir)
        gir_json_cstr = self.ffi.new("char[]", bytes(gir_json, encoding='utf-8'))
        self.inner_graph = self.Clib.initialize_graph(gir_json_cstr)
        HELPER_LOG.debug("initialized inner graph")

    def output_inner_graph(self) -> (GraphIR, Dict[str, str]):
        """
        return the GIR that deserializes from inner egraph, and the name map of output tensors,
        which indicates each output tensor of original GIR is mapping to which output tensor of the output GIR.
        """
        output_json_c_type = self.ffi.new("char[1000000]", b"")
        self.Clib.output_self_graph(self.inner_graph, output_json_c_type)
        deserialized_json: str = self.ffi.string(output_json_c_type).decode("utf-8")
        # HELPER_LOG.debug(f"deserialized json: {deserialized_json}")
        new_inner_graph, graph_output_map = self.trans_json_to_gir(deserialized_json)
        return new_inner_graph, graph_output_map

    def run_saturation(self):
        self.Clib.run_saturation(self.inner_graph)

    def randomly_generate_an_equivalent_graph(self) -> (GraphIR, Dict[str, str]):
        output_json_c_type = self.ffi.new("char[100000]", b"")
        self.Clib.randomly_find_an_equivalent_graph(self.inner_graph, output_json_c_type)
        deserialized_json: str = self.ffi.string(output_json_c_type).decode("utf-8")
        # print(f"random len: {len(deserialized_json)}")
        if deserialized_json == "":
            return None, None
        # HELPER_LOG.debug(f"deserialized json: {deserialized_json}")
        new_inner_graph, graph_output_map = self.trans_json_to_gir(deserialized_json)
        return new_inner_graph, graph_output_map

    def find_the_most_simplified_equivalent_graph(self) -> (GraphIR, Dict[str, str]):
        output_json_c_type = self.ffi.new("char[100000]", b"")
        self.Clib.find_the_most_simplified_equivalent_graph(self.inner_graph, output_json_c_type)
        deserialized_json: str = self.ffi.string(output_json_c_type).decode("utf-8")
        # print(f"simplified len: {len(deserialized_json)}")
        if deserialized_json == "":
            return None, None
        # HELPER_LOG.debug(f"deserialized json: {deserialized_json}")
        new_inner_graph, graph_output_map = self.trans_json_to_gir(deserialized_json)
        return new_inner_graph, graph_output_map

    def find_the_most_complex_equivalent_graph(self) -> (GraphIR, Dict[str, str]):
        output_json_c_type = self.ffi.new("char[100000]", b"")
        self.Clib.find_the_most_complex_equivalent_graph(self.inner_graph, output_json_c_type)
        deserialized_json: str = self.ffi.string(output_json_c_type).decode("utf-8")
        # print(f"complex len: {len(deserialized_json)}")
        if deserialized_json == "":
            return None, None
        # HELPER_LOG.debug(f"deserialized json: {deserialized_json}")
        new_inner_graph, graph_output_map = self.trans_json_to_gir(deserialized_json)
        return new_inner_graph, graph_output_map

    def test_helper_lib(self):
        new_gir, output_map = self.output_inner_graph()
        assert set(output_map.keys()) == set(self.gir.leaf_var()), \
            f"original gir:\n{self.gir.to_dot()}\nnew gir\n{new_gir.to_dot()}\noutput_map:{output_map}"
        assert all([dst_tensor in set(output_map.values()) for dst_tensor in new_gir.leaf_var()]), \
            f"original gir:\n{self.gir.to_dot()}\nnew gir\n{new_gir.to_dot()}\noutput_map:{output_map}"
        self.assert_girs_are_equal(self.gir, new_gir)

        new_gir_2, output_map_2 = self.randomly_generate_an_equivalent_graph()
        assert set(output_map_2.keys()) == set(self.gir.leaf_var()), \
            f"original gir:\n{self.gir.to_dot()}\nnew gir\n{new_gir_2.to_dot()}\noutput_map:{output_map_2}"

    def trans_gir_to_json(self, gir: GraphIR) -> str:
        # tensor_name -> {shape:, dimension:, dtype:}
        vars: Dict[str, Dict] = {}
        # [{op_name: , input_args: , return_values: }, ]
        insts: List[Dict] = []

        for variable_name, tensor in gir.vars.items():  # type: str, AbsTensor
            shape = tensor.shape
            assert isinstance(shape, List)
            dimension = len(shape)
            tensor_dtype = tensor.dtype.__repr__()
            vars[variable_name] = {"shape": deepcopy(shape), "dtype": tensor_dtype}

        for inst in gir.insts:  # type: InstIR
            # input
            input_args = []
            for arg in inst.iexpr.args:
                input_args.append(arg)

            # output
            ret_vals = []
            for retval in inst.retvals():
                ret_vals.append(retval)

            # op
            op_name = self.op_translate_to_str(inst.iexpr.op)

            insts.append({"op": op_name, "input_args": input_args, "return_values": ret_vals, "attributes": {}})

        # deal with output tensor

        for leaf_node in gir.leaf_var():
            insts.append(
                {"op": {"GraphOutput": {"var_name": leaf_node}},
                 "input_args": [leaf_node],
                 "return_values": [],
                 "attributes": {}}
            )

        graph_json = {"vars": vars, "insts": insts}
        x = jsonpickle.encode(graph_json, separators=(',', ':'), use_base85=False)
        HELPER_LOG.debug(f"serialized json: {x}")

        return x

    def trans_json_to_gir(self, json_str: str) -> (GraphIR, Dict[str, str]):
        graph_json = jsonpickle.decode(json_str)
        variables = graph_json["vars"]
        insts = graph_json["insts"]

        # src_var_name -> dst_var_name
        graph_output_map = {}

        vars_for_gir: Dict[str, AbsTensor] = {}
        insts_for_gir: List[InstIR] = []
        gir = GraphIR()

        # deal with vars
        for tensor_name, val in variables.items():
            shape = val["shape"]
            tensor_dtype = val["dtype"]
            tensor = AbsTensor(shape, tensor_dtype)
            vars_for_gir[tensor_name] = tensor

        # deal with insts
        for inst_index, inner_inst in enumerate(insts):  # type: int, Dict
            op_name_or_dic = inner_inst["op"]
            input_args = inner_inst["input_args"]
            return_values = inner_inst["return_values"]
            op = None
            if isinstance(op_name_or_dic, str):
                match op_name_or_dic:
                    case "ReLU":
                        assert len(input_args) == 1
                        assert len(return_values) == 1
                        op = ReLU()
                    case "Sigmoid":
                        assert len(input_args) == 1
                        assert len(return_values) == 1
                        op = Sigmoid()
                    case "Tanh":
                        assert len(input_args) == 1
                        assert len(return_values) == 1
                        op = Tan()
                    case "Add":
                        assert len(input_args) == 2
                        assert len(return_values) == 1
                        op = Add()
                    case "Mul":
                        assert len(input_args) == 2
                        assert len(return_values) == 1
                        op = Mul()
                    case "MatMul":
                        assert len(input_args) == 2
                        assert len(return_values) == 1
                        op = MatMul()
                    case _:
                        raise UnreachableException()
            elif isinstance(op_name_or_dic, Dict):
                assert len(op_name_or_dic.keys()) == 1
                op_name = list(op_name_or_dic.keys())[0]
                op_attrs = op_name_or_dic[op_name]
                match op_name:
                    case "Input":
                        assert len(input_args) == 0
                        assert len(return_values) == 1
                        op = Input(len(return_values[0]))
                        op.extra_attrs["op_index"] = op_attrs["op_index"]
                        op.abs_tensor = vars_for_gir[return_values[0]]
                    case "Constant":
                        assert len(input_args) == 0
                        assert len(return_values) == 1
                        op = Constant(len(return_values[0]))
                        op.extra_attrs["op_index"] = op_attrs["op_index"]
                        op.abs_tensor = vars_for_gir[return_values[0]]
                    case "Transpose":
                        assert len(input_args) == 1
                        assert len(return_values) == 1
                        op = Transpose()
                        op.extra_attrs["dim0"] = op_attrs["dim0"]
                        op.extra_attrs["dim1"] = op_attrs["dim1"]
                    case "MaxPool2d":
                        assert len(input_args) == 1
                        assert len(return_values) == 1
                        op = MaxPool2d(kh=op_attrs["kh"], kw=op_attrs["kw"],
                                       stride=op_attrs["stride"], padding=op_attrs["padding"])
                    case "AvgPool2d":
                        assert len(input_args) == 1
                        assert len(return_values) == 1
                        op = AvgPool2d(kh=op_attrs["kh"], kw=op_attrs["kw"],
                                       stride=op_attrs["stride"], padding=op_attrs["padding"])
                    case "Concat2":
                        assert len(input_args) == 2
                        assert len(return_values) == 1
                        op = Concat2()
                        op.extra_attrs["axis"] = op_attrs["axis"]
                    case "Split2":
                        assert len(input_args) == 1
                        assert len(return_values) == 2
                        op = Split2()
                        op.extra_attrs["axis"] = op_attrs["axis"]
                    case "Other":
                        op = self.op_info[op_attrs["op_index"]]
                    case "GraphOutput":
                        assert len(input_args) == 1
                        assert len(return_values) == 0
                        src_var_name = op_attrs["var_name"]
                        dst_var_name = input_args[0]
                        assert src_var_name not in graph_output_map
                        graph_output_map[src_var_name] = dst_var_name
                        continue
                    case _:
                        raise UnreachableException()
            else:
                raise UnreachableException()

            assert op is not None

            iexpr = InstExpr(op=op, args=input_args)
            inst_ir = InstIR(iexpr=iexpr, identifier=inst_index, irctx=None)
            insts_for_gir.append(inst_ir)

        # assemble gir and maintain the user relations of insts
        gir.vars = vars_for_gir
        gir.insts = insts_for_gir

        # update users
        for inst in gir.insts:  # type InstIR
            for arg in set(inst.iexpr.args):
                inst_id, ret_idx = InstIR.var_inst_idx(arg)
                for idx, may_prod in enumerate(gir.insts):
                    if inst_id == may_prod.identifier:
                        may_prod.users[ret_idx].add(inst)
                        break

        # bind input_like and output_like
        for inst in gir.insts:  # type InstIR
            op = inst.iexpr.op
            itensors = [gir.vars[vname] for vname in inst.iexpr.args]
            otensors = op.checked_type_transfer(itensors)
            op.bind_input_like(itensors)
            op.bind_output_like(otensors)

        gir.assert_wellform()
        return gir, graph_output_map

    @staticmethod
    def assert_girs_are_equal(ir1: GraphIR, ir2: GraphIR):

        def obtain_ir_set(ir: GraphIR, ir_set: Set):
            for inst in ir.insts:
                input_args_str = []
                for arg in inst.iexpr.args:
                    arg_str = ir.vars[arg].__repr__()
                    input_args_str.append(arg_str)
                output_args_str = []
                for arg in inst.retvals():
                    arg_str = ir.vars[arg].__repr__()
                    output_args_str.append(arg_str)
                inst_str = f"{inst.iexpr.op.__class__}({inst.iexpr.op.input_like}, {inst.iexpr.op.output_like}): " \
                           f"{input_args_str}, {output_args_str}"
                ir_set.add(inst_str)

        ir1_set = set()
        ir2_set = set()
        obtain_ir_set(ir1, ir1_set)
        obtain_ir_set(ir2, ir2_set)

        if len(ir1.insts) != 0:
            assert len(ir1_set) != 0, len(ir1_set)
        assert ir1_set == ir2_set, f"ir1: {ir1_set}; ir2: {ir2_set}; " \
                                   f"r1-r2: {ir1_set - ir2_set}; r2-r1: {ir2_set - ir1_set}"

    def op_translate_to_str(self, op: Union[AbsOpBase, Placeholder]) -> str:

        original_op_name: str = op.__repr__()
        assert original_op_name != "Placeholder", f"assert fail: {op}, {original_op_name}"

        op_name = None
        if original_op_name in ["Input", "Constant"]:
            op_index: int = len(self.op_info)
            op.extra_attrs["op_index"] = op_index
            self.op_info.append(op)
            op_name = {original_op_name: {"op_index": op_index}}
        elif original_op_name in ["ReLU", "Sigmoid", "Tanh", "Add", "Mul",
                                  "MatMul"]:
            # important operators that do not use extra attributes
            op_name = original_op_name
        elif original_op_name == "Transpose":
            assert isinstance(op, Transpose)
            op_name = {original_op_name: {"dim0": op.extra_attrs['dim0'],
                                          "dim1": op.extra_attrs['dim1']}}
        # elif original_op_name == "Conv2d":
        #     assert isinstance(inst.iexpr.op, NCHWConv2d)
        #     op_name = {original_op_name: {"in_channels": inst.iexpr.op.in_channels,
        #                                   "out_channels": inst.iexpr.op.out_channels,
        #                                   "kernel_h_size": inst.iexpr.op.kernel_h_size,
        #                                   "kernel_w_size": inst.iexpr.op.kernel_w_size,
        #                                   "stride": inst.iexpr.op.stride, "padding": inst.iexpr.op.padding,
        #                                   "dilation_h": inst.iexpr.op.dilation_h,
        #                                   "dilation_w": inst.iexpr.op.dilation_w}}
        elif original_op_name == "MaxPool2d":
            assert isinstance(op, MaxPool2d)
            op_name = {original_op_name: {"kh": op.kh,
                                          "kw": op.kw,
                                          "stride": op.stride,
                                          "padding": op.padding}}
        elif original_op_name == "AvgPool2d":
            assert isinstance(op, AvgPool2d)
            op_name = {original_op_name: {"kh": op.kh,
                                          "kw": op.kw,
                                          "stride": op.stride,
                                          "padding": op.padding}}
        elif original_op_name == "Concat2":
            assert isinstance(op, Concat2)
            op_name = {original_op_name: {"axis": op.extra_attrs['axis']}}
        elif original_op_name == "Split2":
            assert isinstance(op, Split2)
            op_name = {original_op_name: {"axis": op.extra_attrs['axis']}}
        else:
            # unimportant operators, record op and pass index to the other side
            op_index: int = len(self.op_info)
            op.extra_attrs["op_index"] = op_index
            self.op_info.append(op)
            op_name = {"Other": {"op_index": op_index}}
        assert op_name is not None

        return op_name

    @staticmethod
    def str_translate_to_op(op_name: str) -> Union[AbsOpBase, Placeholder]:
        pass
