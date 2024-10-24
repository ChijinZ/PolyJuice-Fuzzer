from typing import List

import onnx
from pyinfinitensor.onnx import OnnxStub
from multipledispatch import dispatch

from nnsmith.backends import BackendFactory
from nnsmith.backends.factory import BackendCallable
from nnsmith.macro import NNSMITH_ORT_INTRA_OP_THREAD
from nnsmith.materialize.onnx import ONNXModel
from nnsmith.abstract.dtype import DType

class IT(BackendFactory):
    def __init__(self, target, optmax, **kwargs):
        """opt_level ranges from 0 to 3, stands for ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED and ORT_ENABLE_ALL.
        See https://onnxruntime.ai/docs/performance/graph-optimizations.html for detail
        """
        super().__init__(target, optmax, **kwargs)
        
        if target == "cuda":
            from pyinfinitensor.onnx import backend
            self.runtime = backend.cuda_runtime()
        elif target == "cpu":
            from pyinfinitensor import backend
            self.runtime = backend.cpu_runtime()
        else:
            raise ValueError(
                f"Unknown target `{target}`. Only `cpu` and `cuda` are supported."
            )

    @property
    def system_name(self) -> str:
        return "infinitetensor"

    @property
    def import_libs(self) -> List[str]:
        return ["from pyinfinitensor.onnx import OnnxStub", "from pyinfinitensor import backend"]

    @classmethod
    def skip_dtypes(cls) -> List[DType]:
        # TRT will truncate f64 -> f32 and i64 -> i32
        return [DType.float64, DType.float16, DType.uint64, DType.uint16, DType.uint8]

    @dispatch(ONNXModel)
    def make_backend(
        self,
        model: ONNXModel,
    ) -> BackendCallable:
        onnx_model = model.native_model
        stub = OnnxStub(onnx_model, self.runtime)

        out_names = list(model.output_like.keys())
        
        def closure(inputs):
            
            for name, tensor in stub.inputs.items():
                tmp_input = inputs[name]
                tensor.copyin_numpy(tmp_input)

            stub.run()
            res = [v.copyout_numpy() for v in stub.outputs.values()]
            return {n: r for n, r in zip(out_names, res)}

        return closure
