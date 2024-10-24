from dataclasses import dataclass
from typing import Callable, Dict, List, cast

import tensorflow as tf

from nnsmith.abstract.op import AbsOpBase, Input, Constant
from nnsmith.error import SanityCheck
from nnsmith.gir import GraphIR
from nnsmith.logging import TF_LOG
from nnsmith.materialize.tensorflow.forward import forward_fn


@dataclass
class Instr:
    fwd_fn: Callable
    inp_keys: List[str]
    out_keys: List[str]


class TFNet(tf.Module):
    """A TensorFlow network whose computation is defined by a GraphIR."""

    def __init__(self, ir: GraphIR) -> None:
        """Build a TensorFlow model from GraphIR
        Args:
            ir (GraphIR): minimal information for constructing a concrete graph.
        """
        super().__init__()
        self.ir: GraphIR = ir
        self.mlist: List[Callable] = []
        self.instructions: List[Instr] = []
        self.special_op: Dict[int, tf.Module] = {}
        self.input_name_map: Dict[int, str] = {}
        self.constant_record: List[str] = []

        for inst in self.ir.insts:
            if not isinstance(inst.iexpr.op, Input):
                op = cast(AbsOpBase, inst.iexpr.op)
                fwd_fn = forward_fn(op)
                SanityCheck.true(fwd_fn is not None, f"Bad impl for {inst.iexpr.op}")
                if isinstance(fwd_fn, tf.Module):
                    self.mlist.append(fwd_fn)  # Add tf.Module to track its parameters
                instruction_index = len(self.instructions)
                self.instructions.append(Instr(fwd_fn, inst.iexpr.args, inst.retvals()))

                if "op_index" in inst.iexpr.op.extra_attrs:
                    self.special_op[inst.iexpr.op.extra_attrs["op_index"]] = instruction_index
                    if isinstance(op, Constant):
                        self.constant_record.append(inst.iexpr.op.extra_attrs["op_index"])

            else:
                if "op_index" in inst.iexpr.op.extra_attrs:
                    op_index = inst.iexpr.op.extra_attrs["op_index"]
                    assert len(inst.retvals()) == 1
                    self.input_name_map[op_index] = inst.retvals()[0]

    @tf.function(autograph=False)  # disabling autograph makes it faster
    def __call__(self, *args, **kwargs) -> Dict[str, tf.Tensor]:
        return self.__forward(*args, **kwargs)

    @tf.function(autograph=False)
    def call_by_dict(self, x: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        return self.__forward(**x)

    def __forward(self, *args, **kwargs) -> Dict[str, tf.Tensor]:
        mode = "Running Eagerly" if tf.executing_eagerly() else "Tracing"
        TF_LOG.debug(f"{mode} with JIT config: {tf.config.optimizer.get_jit()}")

        key2tensor: Dict[str, tf.Tensor] = {}
        if len(args) == len(self.ir.input_var()):
            for i, key in enumerate(self.ir.input_var()):
                key2tensor[key] = args[i]
        elif len(kwargs) == len(self.ir.input_var()):
            for i, key in enumerate(self.ir.input_var()):
                key2tensor[key] = kwargs[key]
        else:
            raise ValueError("Use either args or kwargs only")

        for instr in self.instructions:
            # get inputs
            inp_tensors = [key2tensor[key] for key in instr.inp_keys]

            # forward
            out_tensors = instr.fwd_fn(*inp_tensors)

            if isinstance(out_tensors, tf.Tensor):
                out_tensors = [out_tensors]
            if isinstance(out_tensors, tuple):
                out_tensors = list(out_tensors)

            # store outputs
            for i_out, out_key in enumerate(instr.out_keys):
                key2tensor[out_key] = out_tensors[i_out]

        return {k: key2tensor[k] for k in self.ir.leaf_var()}
