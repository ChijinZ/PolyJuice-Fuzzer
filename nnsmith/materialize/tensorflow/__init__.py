from __future__ import annotations

import logging
import os
import pickle
from abc import ABC
from os import PathLike
from typing import Callable, Dict, List, Type, Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").disabled = True

import numpy as np
import tensorflow as tf  # type: ignore
from multipledispatch import dispatch  # type: ignore
import random

cuda_string: Optional[str] = None


def get_cuda_string() -> str:
    global cuda_string
    gpus = tf.config.list_logical_devices("GPU")
    assert gpus, "No GPU available"

    if cuda_string is None:
        cuda_string = gpus[random.randint(0, len(gpus) - 1)].name
    return cuda_string


def configure_tf_gpu_mem(max_megabytes=None):
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
        if isinstance(max_megabytes, int):
            tf.config.set_logical_device_configuration(
                gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=max_megabytes)]
            )


configure_tf_gpu_mem()

from nnsmith.abstract.op import AbsOpBase, AbsTensor
from nnsmith.gir import GraphIR
from nnsmith.materialize import Model, Oracle
from nnsmith.materialize.tensorflow.forward import ALL_TF_OPS
from nnsmith.materialize.tensorflow.tfnet import TFNet, Instr
from nnsmith.util import register_seed_setter

TFNetCallable = Callable[..., Dict[str, tf.Tensor]]


@dispatch(dict)
def randn_from_specs(specs: Dict[str, tf.TensorSpec]) -> Dict[str, tf.Tensor]:
    return {
        name: tf.cast(tf.random.normal(shape=spec.shape), dtype=spec.dtype)
        for name, spec in specs.items()
    }


def np_dict_from_tf(x: Dict[str, tf.Tensor]) -> Dict[str, np.ndarray]:
    return {key: np.array(value.numpy()) for key, value in x.items()}


def tf_dict_from_np(x: Dict[str, np.ndarray]) -> Dict[str, tf.Tensor]:
    return {key: tf.convert_to_tensor(value) for key, value in x.items()}


class EagerModeCtx:
    def __init__(self, eagerly: bool) -> None:
        assert isinstance(
            eagerly, bool
        ), f"argument eagerly should not be {eagerly.__class__}. It must be a boolean."
        self.eagerly = eagerly

    def __enter__(self) -> None:
        self.old_mode = tf.config.functions_run_eagerly()
        tf.config.run_functions_eagerly(self.eagerly)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        tf.config.run_functions_eagerly(self.old_mode)


class TFModel(Model, ABC):  # Don't directly instantiate this class
    """Wrapper class of TFNet (tf.Module)
    It only stores net. Other information like I/O info are temporarily generated.
    Other values like I/O values should be managed by the user.
    """

    def __init__(self, ir: GraphIR) -> None:
        super().__init__()
        self.ir = ir
        self.net = TFNet(ir=ir)

    @property
    def gir(self) -> GraphIR:
        return self.ir

    @classmethod
    def from_gir(cls: Type["TFModel"], ir: GraphIR, **kwargs) -> "TFModel":
        return cls(ir)

    @property
    def version(self) -> str:
        return tf.__version__

    @property
    def native_model(self) -> TFNet:
        return self.net

    @property
    def input_like(self) -> Dict[str, AbsTensor]:
        return {ik: self.ir.vars[ik] for ik in self.ir.input_var()}

    @property
    def output_like(self) -> Dict[str, AbsTensor]:
        return {ok: self.ir.vars[ok] for ok in self.ir.leaf_var()}

    @property
    def input_specs(self) -> Dict[str, tf.TensorSpec]:
        ret: Dict[str, tf.TensorSpec] = {}
        for k, aten in self.input_like.items():
            ret[k] = tf.TensorSpec(aten.shape, aten.dtype.tensorflow(), k)
        return ret

    @staticmethod
    def name_suffix() -> str:
        return ""

    def make_oracle(
            self, inputs: Dict[str, tf.Tensor | tf.TensorSpec] = None
    ) -> Oracle:
        assert not self.needs_grad_check(), "grad_check is not supported in TF"

        if inputs is None or isinstance(list(inputs.values())[0], tf.TensorSpec):
            input_dict = self.random_inputs()
        else:
            input_dict = inputs
        output_dict = self.run_eagerly(input_dict)

        if all(isinstance(value, tf.Tensor) for _, value in input_dict.items()):
            input_dict = np_dict_from_tf(input_dict)
        if all(isinstance(value, tf.Tensor) for _, value in output_dict.items()):
            output_dict = np_dict_from_tf(output_dict)

        return Oracle(input_dict, output_dict, provider="tf[cpu] eager")

    def dump(self, path: PathLike = "saved_tfmodel") -> None:
        os.makedirs(path, exist_ok=True)
        # gir.pkl
        with open(os.path.join(path, TFModel.gir_name()), "wb") as f:
            pickle.dump(self.ir, f)
        # tfnet
        with self.device:
            concrete_net = self.concrete_net(self.input_specs)
            tf.saved_model.save(
                self.net,
                os.path.join(path, TFModel.tfnet_dir_name()),
                signatures=concrete_net,
            )

    def dump_with_oracle(
            self,
            path: PathLike,
            inputs: Dict[str, tf.Tensor | tf.TensorSpec] = None,
    ) -> None:
        self.dump(path)
        oracle = self.make_oracle(inputs)
        oracle.dump(os.path.join(path, Oracle.name()))

    @classmethod
    def load(cls, path: PathLike) -> "TFModel":
        with open(os.path.join(path, cls.gir_name()), "rb") as f:
            gir: GraphIR = pickle.load(f)
        model = cls(gir)
        model.net = tf.saved_model.load(os.path.join(path, TFModel.tfnet_dir_name()))
        return model

    @staticmethod
    def gir_name() -> str:
        return "gir.pkl"

    @staticmethod
    def in_out_pkl_name():
        return "in_out.pkl"

    @staticmethod
    def tfnet_dir_name():
        return "tfnet"

    def random_inputs(self) -> Dict[str, tf.Tensor]:
        return {
            spec.name: tf.cast(
                tf.random.uniform(
                    shape=spec.shape,
                    minval=-100.0,
                    maxval=100.0,
                    seed=None,
                ),
                dtype=spec.dtype,
            )
            for spec in self.input_specs.values()
        }

    def concrete_net(
            self, inputs: Dict[str, tf.Tensor | tf.TensorSpec] | None = None
    ) -> TFNetCallable:
        if inputs is None:
            inputs = self.input_specs
        return self.net.__call__.get_concrete_function(**inputs)

    def refine_weights(self) -> None:
        pass

    def run_eagerly(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        with EagerModeCtx(True), self.device:  # disable graph execution
            # TODO some op can only run on GPU (e.g. conv with NCHW)
            results = self.net(**inputs)
        return results

    @staticmethod
    def operators() -> List[Type[AbsOpBase]]:
        return list(ALL_TF_OPS)

    @property
    def import_libs(self) -> List[str]:
        return ["import tensorflow as tf"]

    @staticmethod
    def add_seed_setter() -> None:
        register_seed_setter("tensorflow", tf.random.set_seed, overwrite=True)

    @staticmethod
    def const(data):
        return lambda: tf.constant(data)

    def make_weights_consist_with_other(self, other_model: Model):
        assert isinstance(other_model, TFModel)
        assert set(self.net.special_op.keys()) == set(other_model.net.special_op.keys())
        for op_index in self.net.special_op.keys():
            self_op = self.net.instructions[self.net.special_op[op_index]].fwd_fn
            other_op = other_model.net.instructions[other_model.net.special_op[op_index]].fwd_fn
            assert self_op.__class__ == other_op.__class__
            if isinstance(self_op, tf.Module):
                print(f"tf module index: {op_index}")
                self_op.set_weights(other_op.get_weights())
            elif op_index in self.net.constant_record:
                other_constant = other_op()
                instructions_index = self.net.special_op[op_index]
                instruction = self.net.instructions[instructions_index]
                self.net.instructions[instructions_index] = Instr(self.const(other_constant), instruction.inp_keys,
                                                                  instruction.out_keys)

    def corresponding_input_map(self, other_model: Model) -> Dict[str, str]:
        assert isinstance(other_model, TFModel)
        # assert (self.net.input_name_map.keys()) == set(other_model.net.input_name_map.keys())
        name_map = {}
        for op_index in self.net.input_name_map.keys():
            self_input_name = self.net.input_name_map[op_index]
            other_input_name = other_model.net.input_name_map[op_index]
            name_map[self_input_name] = other_input_name
        return name_map


class TFModelCPU(TFModel):
    @property
    def device(self) -> tf.device:
        return tf.device(tf.config.list_logical_devices("CPU")[0].name)


class TFModelCUDA(TFModel):
    @property
    def device(self) -> tf.device:
        # gpus = tf.config.list_logical_devices("GPU")
        return tf.device(get_cuda_string())
