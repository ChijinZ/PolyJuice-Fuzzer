from dataclasses import dataclass
from typing import List

import numpy as np
import onnx
import pycuda.driver as cuda
import tensorrt as trt
from multipledispatch import dispatch
from pycuda.driver import DeviceAllocation

from nnsmith.abstract.arith import *
from nnsmith.abstract.dtype import DType
from nnsmith.abstract.extension import patch_requires
from nnsmith.abstract.op import AbsOpBase
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.backends import BackendFactory
from nnsmith.materialize.onnx import ONNXModel
from polygraphy.backend.trt import EngineFromNetwork, TrtRunner, NetworkFromOnnxBytes
from polygraphy.logger.logger import G_LOGGER
G_LOGGER.module_severity = G_LOGGER.ERROR


@dataclass
class HostDeviceMem:
    host: np.ndarray
    device: DeviceAllocation


class TRT(BackendFactory):
    def __init__(self, target="cuda", optmax=True, **kwargs):
        super().__init__(target, optmax, **kwargs)

        if target != "cuda":
            raise ValueError("TensorRT backend only supports GPU!")

        if optmax is False:
            # TODO(@ganler): support non-optimized TensorRT by using performing
            # inference over a model that marks all nodes as outputs.
            raise ValueError("There is not O0 mode for TensorRT so far.")

    @property
    def system_name(self) -> str:
        return "tensorrt"

    @property
    def version(self) -> str:
        return trt.__version__

    @dispatch(ONNXModel)
    def make_backend(self, model: ONNXModel):
        engine = EngineFromNetwork(NetworkFromOnnxBytes(model.native_model.SerializeToString()))

        def closure(inputs):
            with TrtRunner(engine) as runner:
                return runner.infer(inputs)

        return closure

    @property
    def import_libs(self) -> List[str]:
        return ["import tensorrt as trt"]

    @classmethod
    def skip_dtypes(cls) -> List[DType]:
        # TRT will truncate f64 -> f32 and i64 -> i32
        return [DType.float64, DType.int64]


@patch_requires(TRT.system_name, "core.Pool2d")
def RulePool2d(self: AbsOpBase, _: List[AbsTensor]) -> List[Union[z3.BoolRef, bool]]:
    return [nnsmith_lt(nnsmith_mul(self.kh, self.kw), 10000)]
