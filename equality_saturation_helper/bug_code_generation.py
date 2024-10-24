from nnsmith.materialize import Render, BugReport, TestCase, Model
from nnsmith.materialize.onnx import ONNXModel
from nnsmith.materialize.torch import TorchModel
from nnsmith.backends import BackendFactory
import argparse
import os
from typing import Dict, List, Optional
import numpy as np
import pickle

TAB = " " * 4


# tensor_dict_name_* is the name of the variable which stores a dict of output name to actual tensor
def diff_two_tensors(tensor_dict_name_0: str,
                     tensor_dict_name_1: str,
                     output_name_map: str,
                     name_of_this_assertion: str) -> str:
    output = ""
    output += "print('=========================')\n"
    output += "try:\n"
    output += f"{TAB}for tensor_name_0, tensor_name_1 in {output_name_map}.items():\n"
    output += f"{TAB}{TAB}testing.assert_allclose(" \
              f"{tensor_dict_name_0}[tensor_name_0], {tensor_dict_name_1}[tensor_name_1], rtol=1, " \
              "err_msg=f'at {tensor_name_0}, {tensor_name_1}')\n"
    output += f"{TAB}print(\"{name_of_this_assertion} does not trigger assertion\")\n"
    output += "except AssertionError as e:\n"
    output += f"{TAB}print(\"{name_of_this_assertion} triggers assertion\")\n"
    output += f"{TAB}print(e)\n"
    output += "print('=========================')\n"

    return output


def onnx_to_tvm(onnx_path: Optional[str] = None,
                input_dict: Optional[Dict] = None,
                data_output_path: Optional[str] = None,
                index: int = 0,
                output_import: bool = False,
                opt_level: int = 4,
                is_single_output: bool = False) -> str:
    if data_output_path is not None and input_dict is not None:
        with open(data_output_path, "wb+") as f:
            pickle.dump(input_dict, f)
    # dtype = input_dict[list(input_dict.keys())[0]].dtype
    output = ""
    if output_import:
        output += "\n".join(
            ["import tvm",
             "import onnx",
             "from tvm import relay",
             "import numpy as np",
             "import pickle",
             "from numpy import testing",
             "import onnxruntime as ort",
             "import torch"]
        )
        output += "\n\n"
    if onnx_path:
        output += f"onnx_model_{index} = onnx.load('{onnx_path}')\n"
        output += f"onnx_model_outputs_{index} = [node.name for node in onnx_model_{index}.graph.output]\n"
    if data_output_path is not None:
        output += f"input_dict_{index} = pickle.load(open('{data_output_path}', 'rb'))\n"
    # output += f"input_dict = {{key: np.array(val, dtype='{dtype}') for key, val in input_dict.items()}}\n"
    output += f"shape_dict_{index} = {{key: val.shape for key, val in input_dict_{index}.items()}}\n"
    output += f"mod_{index}, params_{index} = " \
              f"relay.frontend.from_onnx(onnx_model_{index}, shape_dict_{index}, freeze_params=True)\n"
    output += f"with tvm.transform.PassContext(opt_level={opt_level}):\n"
    output += f"{TAB}executor_{index} = relay.build_module.create_executor(\"graph\", mod_{index}, tvm.cpu(), " \
              f"tvm.target.Target(\"llvm\"), params_{index}).evaluate()\n"
    if is_single_output:
        output += f"{TAB}executor_res_{index} = [executor_{index}(**input_dict_{index}).numpy()]\n"
    else:
        output += f"{TAB}executor_res_{index} = [tensor.numpy() for tensor in executor_{index}(**input_dict_{index})]\n"

    output += f"{TAB}output_{index} = dict(" \
              f"zip(onnx_model_outputs_{index}, executor_res_{index}))\n"

    return output


def onnx_to_torch(onnx_path: str, input_dict: Dict, data_output_path: str, use_pt_compile: bool = False) -> str:
    with open(data_output_path, "wb+") as f:
        pickle.dump(input_dict, f)
    output = "\n".join(
        ["from onnx2torch import convert", "import onnx", "import torch", "import numpy as np", "import pickle"])
    output += "\n"
    output += f"onnx_model = onnx.load('{onnx_path}')\n"
    output += f"input_dict = pickle.load(open('{data_output_path}', 'rb'))\n"
    # output += f"input_dict = {{key: np.array(val, dtype='{dtype}') for key, val in input_dict.items()}}\n"
    output += "torch_model = convert(onnx_model)\n"
    if use_pt_compile:
        output += "torch_model = torch.compile(torch_model)\n"
    output += "res = torch_model(*[torch.from_numpy(v) for _, v in input_dict.items()])\n"
    output += "print(res)"
    return output


def onnx_to_onnxruntime(output_names: List[str],
                        onnx_path: Optional[str] = None,
                        input_dict: Optional[Dict] = None,
                        data_output_path: Optional[str] = None,
                        index: int = 0,
                        output_import: bool = False,
                        enable_opt: bool = True) -> str:
    if data_output_path is not None and input_dict is not None:
        with open(data_output_path, "wb+") as f:
            pickle.dump(input_dict, f)

    graph_optimization_level = "ort.GraphOptimizationLevel.ORT_ENABLE_ALL" \
        if enable_opt else "ort.GraphOptimizationLevel.ORT_DISABLE_ALL"

    output = ""
    if output_import:
        output += "\n".join([
            "import onnxruntime as ort",
            "import onnx",
            "import numpy as np",
            "import pickle",
            "from numpy import testing",
            "import tvm",
            "from tvm import relay",
            "import torch"]
        )
        output += "\n\n"
    # output += f"onnx_model = onnx.load('{onnx_path}')\n"
    if data_output_path is not None:
        output += f"input_dict_{index} = pickle.load(open('{data_output_path}', 'rb'))\n"
    # output += f"output_names_{index} = {output_names}\n"
    output += f"sess_options_{index} = ort.SessionOptions()\n"
    output += f"sess_options_{index}.graph_optimization_level = {graph_optimization_level}\n"
    output += f"sess_{index} = ort.InferenceSession('{onnx_path}'," \
              f"providers=['CPUExecutionProvider']," \
              f"sess_options=sess_options_{index})\n"
    output += f"sess_res_{index} = sess_{index}.run(output_names_{index}, input_dict_{index})\n"
    output += f"output_{index} = dict(zip(output_names_{index}, sess_res_{index}))\n"

    return output

def onnx_to_tensorrt(onnx_path: str,
                     data_output_path: str,
                     index: int = 0) -> str:
    output = ""
    output += f"engine_{index} = EngineFromNetwork(NetworkFromOnnxPath('{onnx_path}'))\n"
    output += f"with TrtRunner(engine_{index}) as runner:\n"
    output += f"{TAB}output_{index} = runner.infer(input_dict_{index})\n"

    return output

def onnx_to_infinitensor(onnx_path: Optional[str] = None,
                   index: int = 0,
                   device: str = None):
    output = ""
    output += f"stub_{index} = OnnxStub(onnx.load('{onnx_path}'), backend.{device}_runtime())\n"
    output += f"for name, tensor in stub_{index}.inputs.items():\n"
    output += f"{TAB}tensor.copyin_numpy(input_dict_{index}[name])\n"
    output += f"stub_{index}.run()\n"
    output += f"output_{index} = dict(zip(output_names_{index}, [v.copyout_numpy() for v in stub_{index}.outputs.values()]))\n"

    return output

def torch_to_torch_compile(model: Optional[TorchModel] = None,
                           input_dict: Optional[Dict] = None,
                           data_output_path: Optional[str] = None,
                           index: int = 0,
                           enable_opt: bool = True) -> str:
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    if model:
        render = Render()
        render.emit_model(model)
        with open(data_output_path, "wb+") as f:
            pickle.dump(input_dict, f)
        render.emit_input(model, path=data_output_path)
        render.emit_backend(BackendFactory.init("pt2"))
        return render
    
    if data_output_path is not None and input_dict is not None:
        with open(data_output_path, "wb+") as f:
            pickle.dump(input_dict, f)

    output = ""
    # for key, val in model.native_model.state_dict().items():
    #     val_str = str(val.numpy()).replace("\n", ",")
    #     print(val.shape)
    #     print(val_str)
    #     output += f"# self.{key} = torch.nn.Parameter(torch.tensor([{val_str}], dtype={val.dtype}), requires_grad=False)\n"

    if data_output_path is not None:
        output += f"input_data_{index} = [v for _, v in pickle.load(open('{data_output_path}', 'rb')).items()]\n\n"
    
    if enable_opt:
        output += f"optmodel_{index} = torch.compile(model_{index}, fullgraph=True, backend='inductor', mode=None)\n"
        output += f"model_out_{index} = optmodel_{index}(*[torch.from_numpy(v).to(DEVICE) for v in input_data_{index}])\n"
    else:
        output += f"model_out_{index} = model_{index}(*[torch.from_numpy(v).to(DEVICE) for v in input_data_{index}])\n"
    output += f"model_out_{index} = [v.to(DEVICE).detach() for v in model_out_{index}] if isinstance(model_out_{index}, tuple) else [model_out_{index}.to(DEVICE).detach()]\n"
    output += f"model_out_{index} = [v.cpu().resolve_conj().numpy() if v.is_conj() else v.cpu().numpy() for v in model_out_{index}]\n"
    output += f"output_{index} = dict(zip(output_names_{index}, model_out_{index}))\n"

    return output

def torch_to_hidet(model: Optional[TorchModel] = None,
                   input_dict: Optional[Dict] = None,
                   data_output_path: Optional[str] = None,
                   index: int = 0,
                   enable_opt: bool = True) -> str:
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    if model:
        render = Render()
        render.emit_model(model)
        with open(data_output_path, "wb+") as f:
            pickle.dump(input_dict, f)
        render.emit_input(model, path=data_output_path)
        render.emit_backend(BackendFactory.init("pt2"))
        return render
    
    if data_output_path is not None and input_dict is not None:
        with open(data_output_path, "wb+") as f:
            pickle.dump(input_dict, f)

    output = ""
    # for key, val in model.native_model.state_dict().items():
    #     val_str = str(val.numpy()).replace("\n", ",")
    #     print(val.shape)
    #     print(val_str)
    #     output += f"# self.{key} = torch.nn.Parameter(torch.tensor([{val_str}], dtype={val.dtype}), requires_grad=False)\n"

    if data_output_path is not None:
        output += f"input_data_{index} = [v for _, v in pickle.load(open('{data_output_path}', 'rb')).items()]\n\n"
    
    if enable_opt:
        output += f"optmodel_{index} = torch.compile(model_{index}, fullgraph=True, backend='hidet', mode=None)\n"
        output += f"model_out_{index} = optmodel_{index}(*[torch.from_numpy(v).to(DEVICE) for v in input_data_{index}])\n"
    else:
        output += f"model_out_{index} = model_{index}(*[torch.from_numpy(v).to(DEVICE) for v in input_data_{index}])\n"
    output += f"model_out_{index} = [v.to(DEVICE).detach() for v in model_out_{index}] if isinstance(model_out_{index}, tuple) else [model_out_{index}.to(DEVICE).detach()]\n"
    output += f"model_out_{index} = [v.cpu().resolve_conj().numpy() if v.is_conj() else v.cpu().numpy() for v in model_out_{index}]\n"
    output += f"output_{index} = dict(zip(output_names_{index}, model_out_{index}))\n"

    return output

def tensorflow_to_xla(tf_model_path: Optional[str] = None,
                      input_dict: Optional[Dict] = None,
                      data_output_path: Optional[str] = None,
                      index: int = 0,
                      output_import: bool = False,
                      enable_opt: bool = False,
                      device: Optional[str] = None) -> str:
    if data_output_path is not None and input_dict is not None:
        with open(data_output_path, "wb+") as f:
            pickle.dump(input_dict, f)
    
    output = ""
    if output_import:
        output += "\n".join([
            "import tensorflow as tf",
            "import numpy as np",
            "import pickle",
            "from numpy import testing",
            "from typing import Dict\n",
            "def tf_dict_from_np(x: Dict[str, np.ndarray]) -> Dict[str, tf.Tensor]:",
            TAB+"return {key: tf.convert_to_tensor(value) for key, value in x.items()}\n",
            "def np_dict_from_tf(x: Dict[str, tf.Tensor]) -> Dict[str, np.ndarray]:",
            TAB+"return {key: np.array(value.numpy()) for key, value in x.items()}\n"
            ]
        )
        output += "\n"

    if device and device == "cpu":
        output += "device = tf.device(tf.config.list_logical_devices('CPU')[0].name)\n"
        output += "\n"
    if tf_model_path:
        output += f"model_{index} = tf.saved_model.load('{tf_model_path}')\n"
    if data_output_path:
        output += f"input_dict_{index} = tf_dict_from_np(pickle.load(open('{data_output_path}', 'rb')))\n"
    if enable_opt:
        output += f"xla_model_{index} = tf.function(model_{index}, jit_compile=True)\n"
        output += f"with device:\n"
        output += f"{TAB}output_{index} = np_dict_from_tf(xla_model_{index}(**input_dict_{index}))\n"
    else:
        output += f"with device:\n"
        output += f"{TAB}output_{index} = np_dict_from_tf(model_{index}(**input_dict_{index}))\n"
    return output


def torch_to_torch_code(model: TorchModel, 
                        index: int = 0,
                        output_import: bool = False,
                        device: Optional[str] = None,) -> str:
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    output = ""
    if output_import:
        output += "\n".join([
            "import numpy as np",
            "import pickle",
            "from numpy import testing",
            "import torch"]
        )
        output += "\n\n"

    if device:
        output += f"DEVICE='{device}'\n\n"

    output_names = list(model.output_like.keys())
    output += model.emit_def(mod_name=f"model_{index}", mod_cls=f"Model{index}")
    output += f"output_names_{index} = {output_names}\n"

    return output

def onnx_to_torch_code(model: ONNXModel, 
                          input_dict: Dict, 
                          data_output_path: str,
                          index: int = 0,
                          output_import: bool = False,
                          device: Optional[str] = None) -> str:
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    with open(data_output_path, "wb+") as f:
        pickle.dump(input_dict, f)
    
    output = ""
    if output_import:
        output += "\n".join([
            "import onnxruntime as ort",
            "import onnx",
            "import numpy as np",
            "import pickle",
            "from numpy import testing",
            "import tvm",
            "from tvm import relay",
            "import torch"]
        )
        output += "\n\n"

    if device:
        output += f"DEVICE='{device}'\n\n"
    # for key, val in model.torch_model.native_model.state_dict().items():
    #     val_str = str(val.numpy()).replace("\n", ",")
    #     print(val.shape)
    #     print(val_str)
    #     output += f"# self.{key} = torch.nn.Parameter(torch.tensor([{val_str}], dtype={val.dtype}), requires_grad=False)\n"

    output_names = list(model.output_like.keys())
    input_names = list(input_dict.keys())

    output += model.emit_def(mod_name=f"model_{index}", mod_cls=f"Model{index}")
    # output += f"model_{index}.eval()\n"
    output += f"output_names_{index} = {output_names}\n"
    output += f"input_dict_{index} = pickle.load(open('{data_output_path}', 'rb'))\n"
    output += f"inputs_{index} = tuple(torch.from_numpy(v).to(DEVICE) for _, v in input_dict_{index}.items())\n"
    output += f"torch.onnx.export(model_{index}, inputs_{index}, '{index}.onnx', verbose=False, input_names={input_names}, output_names=output_names_{index}, opset_version=14, do_constant_folding=False)\n"
    
    return output

def generation_torch_to_torch_compile(model_0: TorchModel, 
                                      model_1: TorchModel, 
                                      input_dict_0: Optional[Dict] = None, 
                                      input_dict_1: Optional[Dict] = None, 
                                      output_map: Optional[Dict] = None, 
                                      backend_target: Optional[str] = None):
    output_str = ""
    output_str += torch_to_torch_code(model=model_0, 
                                        index=0, 
                                        output_import=True,
                                        device=backend_target)
    output_str += "\n"
    output_str += torch_to_torch_code(model=model_1, 
                                        index=1, 
                                        output_import=False)
    output_str += "\n"
    output_str += torch_to_torch_compile(
        input_dict=input_dict_0,
        data_output_path=f"{0}.pickle",
        index=0,
        enable_opt=True,
    )
    output_str += "\n"
    output_str += torch_to_torch_compile(
        input_dict=input_dict_1,
        data_output_path=f"{1}.pickle",
        index=1,
        enable_opt=True,
    )
    output_str += f"output_name_dict = {output_map}\n"
    output_str += "\n"
    output_str += diff_two_tensors("output_0", "output_1", "output_name_dict",
                                    "torch_complie")
    
    output_str += "\n"
    output_str += torch_to_torch_compile(
        index=0,
        enable_opt=False,
    )
    output_str += "\n"
    output_str += torch_to_torch_compile(
        index=1,
        enable_opt=False,
    )
    output_str += "\n"
    output_str += diff_two_tensors("output_0", "output_1", "output_name_dict",
                                    "torch_eager")

    return output_str

def generation_torch_to_hidet(model_0: TorchModel, 
                              model_1: TorchModel, 
                              input_dict_0: Optional[Dict] = None, 
                              input_dict_1: Optional[Dict] = None, 
                              output_map: Optional[Dict] = None, 
                              backend_target: Optional[str] = None):
    output_str = ""
    print(backend_target)
    output_str += torch_to_torch_code(model=model_0, 
                                      index=0, 
                                      output_import=True,
                                      device=backend_target)
    output_str += "\n"
    output_str += torch_to_torch_code(model=model_1, 
                                      index=1, 
                                      output_import=False)
    output_str += "\n"
    output_str += torch_to_hidet(
        input_dict=input_dict_0,
        data_output_path=f"{0}.pickle",
        index=0,
        enable_opt=True,
    )
    output_str += "\n"
    output_str += torch_to_hidet(
        input_dict=input_dict_1,
        data_output_path=f"{1}.pickle",
        index=1,
        enable_opt=True,
    )
    output_str += f"output_name_dict = {output_map}\n"
    output_str += "\n"
    output_str += diff_two_tensors("output_0", "output_1", "output_name_dict",
                                    "hidet")
    
    output_str += "\n"
    output_str += torch_to_hidet(
        index=0,
        enable_opt=False,
    )
    output_str += "\n"
    output_str += torch_to_hidet(
        index=1,
        enable_opt=False,
    )
    output_str += "\n"
    output_str += diff_two_tensors("output_0", "output_1", "output_name_dict",
                                    "torch_eager")

    return output_str

def generation_onnx_to_tensorrt(model_0: ONNXModel,
                                model_1: ONNXModel, 
                                input_dict_0: Optional[Dict] = None, 
                                input_dict_1: Optional[Dict] = None, 
                                output_map: Optional[Dict] = None, 
                                backend_target: Optional[str] = None):
    output_str = ""
    output_str += "\n".join([
            "import numpy as np",
            "from numpy import testing",
            "import pickle",
            "import torch",
            "from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath, TrtRunner"]
        )
    output_str += "\n\n"
    output_str += onnx_to_torch_code(model=model_0, 
                                     input_dict=input_dict_0,
                                     data_output_path=f"{0}.pickle", 
                                     index=0, 
                                     output_import=False,
                                     device=backend_target
                                     )
    output_str += "\n"
    output_str += onnx_to_torch_code(model=model_1, 
                                     input_dict=input_dict_1,
                                     data_output_path=f"{1}.pickle", 
                                     index=1, 
                                     output_import=False)
    output_str += "\n"
    output_str += onnx_to_tensorrt(onnx_path=f"{0}.onnx",
                                   data_output_path=f"{0}.pickle",
                                   index=0)
    output_str += "\n"
    output_str += onnx_to_tensorrt(onnx_path=f"{1}.onnx",
                                   data_output_path=f"{1}.pickle",
                                   index=1)
    output_str += f"output_name_dict = {output_map}\n"
    output_str += "\n"
    output_str += diff_two_tensors("output_0", "output_1", "output_name_dict", "tensorRT")
    return output_str
    
def generation_tensorflow_to_xla(model_path_0: str,
                                 model_path_1: str,
                                 input_dict_0: Optional[Dict] = None, 
                                 input_dict_1: Optional[Dict] = None, 
                                 output_map: Optional[Dict] = None, 
                                 backend_target: Optional[str] = None):
    output_str = ""
    output_str += tensorflow_to_xla(tf_model_path=model_path_0,
                                    input_dict=input_dict_0,
                                    data_output_path="0.pickle",
                                    index=0,
                                    output_import=True,
                                    enable_opt=True,
                                    device=backend_target)
    output_str += "\n"
    output_str += tensorflow_to_xla(tf_model_path=model_path_1,
                                    input_dict=input_dict_1,
                                    data_output_path="1.pickle",
                                    index=1,
                                    enable_opt=True)
    output_str += f"output_name_dict = {output_map}\n"
    output_str += "\n"
    output_str += diff_two_tensors("output_0", "output_1", "output_name_dict", "xla")
    output_str += "\n"
    output_str += tensorflow_to_xla(index=0,
                                    enable_opt=False)
    output_str += "\n"
    output_str += tensorflow_to_xla(index=1,
                                    enable_opt=False)
    output_str += "\n"
    output_str += diff_two_tensors("output_0", "output_1", "output_name_dict", "tensorflow")
    return output_str

def generation_onnx_to_infinitensor(model_0: ONNXModel,
                                model_1: ONNXModel, 
                                input_dict_0: Optional[Dict] = None, 
                                input_dict_1: Optional[Dict] = None, 
                                output_map: Optional[Dict] = None, 
                                backend_target: Optional[str] = None):
    output_str = ""
    output_str += "\n".join([
            "import numpy as np",
            "from numpy import testing",
            "import pickle",
            "import onnx",
            "import torch",
            "import onnxruntime as ort",
            "from pyinfinitensor.onnx import OnnxStub, backend"]
        )
    output_str += "\n\n"
    output_str += onnx_to_torch_code(model=model_0, 
                                     input_dict=input_dict_0,
                                     data_output_path=f"{0}.pickle", 
                                     index=0, 
                                     output_import=False,
                                     device=backend_target
                                     )
    output_str += "\n"
    output_str += onnx_to_torch_code(model=model_1, 
                                     input_dict=input_dict_1,
                                     data_output_path=f"{1}.pickle", 
                                     index=1, 
                                     output_import=False)
    output_str += "\n"
    output_str += onnx_to_infinitensor(onnx_path=f"{0}.onnx",
                                 index=0,
                                 device=backend_target)
    output_str += "\n"
    output_str += onnx_to_infinitensor(onnx_path=f"{1}.onnx",
                                 index=1,
                                 device=backend_target)
    output_str += f"output_name_dict = {output_map}\n"
    output_str += "\n"
    output_str += diff_two_tensors("output_0", "output_1", "output_name_dict", "InfiniTensor")
    output_str += "\n"
    output_str += onnx_to_onnxruntime(
        output_names=list(model_0.output_like.keys()),
        onnx_path=f"{0}.onnx",
        index=0,
        output_import=False,
        enable_opt=False
    )
    output_str += "\n"
    output_str += onnx_to_onnxruntime(
        output_names=list(model_1.output_like.keys()),
        onnx_path=f"{1}.onnx",
        index=1,
        output_import=False,
        enable_opt=False
    )
    output_str += "\n"
    output_str += diff_two_tensors("output_0", "output_1", "output_name_dict",
                                    "onnxruntime")
    output_str += "\n"

    return output_str
    

def main():
    parser = argparse.ArgumentParser(description="generate code from a bug report")
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
    bug_report = BugReport.load(model_init, input_path)

    os.makedirs(output_path, exist_ok=True)

    match model_type:
        case "onnx":
            testcases = []
            if bug_report.eq_testcase is not None:
                onnx_path_1 = os.path.join(input_path, "t1", "model.onnx")
                onnx_path_2 = os.path.join(input_path, "t2", "model.onnx")
                input_1 = bug_report.testcase.oracle.input
                input_2 = bug_report.eq_testcase.oracle.input
                # print(input_1)
                # print(input_2)
                testcases = [(onnx_path_1, input_1, bug_report.testcase),
                             (onnx_path_2, input_2, bug_report.eq_testcase)]
            else:
                onnx_path = os.path.join(input_path, "model.onnx")
                input_dict = bug_report.testcase.oracle.input
                testcases = [(onnx_path, input_dict, bug_report.testcase)]
            match desired_code:
                case "tvm":
                    assert len(testcases) <= 2
                    if len(testcases) == 1:
                        output_str = onnx_to_torch_code(model=testcases[0][2].model, 
                                                            input_dict=testcases[0][1],
                                                            data_output_path=os.path.join(output_path, f"{0}.pickle"), 
                                                            index=0, 
                                                            output_import=True,
                                                            device=backend_target)
                        output_str += "\n"
                        output_str += onnx_to_tvm(onnx_path=f"{0}.onnx",
                                                input_dict=testcases[0][1],
                                                index=0,
                                                output_import=False,
                                                opt_level=4,
                                                is_single_output=len(testcases[0][2].model.output_like.keys()) == 1)

                    else:
                        output_str = f"# input dict: " \
                                    f"{bug_report.testcase.model.corresponding_input_map(bug_report.eq_testcase.model)}\n"
                        output_str += "\n"


                        output_str += onnx_to_torch_code(model=testcases[0][2].model, 
                                                            input_dict=testcases[0][1],
                                                            data_output_path=os.path.join(output_path, f"{0}.pickle"), 
                                                            index=0, 
                                                            output_import=True,
                                                            device=backend_target)
                        output_str += "\n"
                        output_str += onnx_to_torch_code(model=testcases[1][2].model, 
                                                            input_dict=testcases[1][1],
                                                            data_output_path=os.path.join(output_path, f"{1}.pickle"), 
                                                            index=1, 
                                                            output_import=False)
                        output_str += "\n"
                        output_str += onnx_to_tvm(onnx_path=f"{0}.onnx",
                                                input_dict=testcases[0][1],
                                                index=0,
                                                output_import=False,
                                                opt_level=4,
                                                is_single_output=len(testcases[0][2].model.output_like.keys()) == 1)
                        output_str += "\n"
                        output_str += onnx_to_tvm(onnx_path=f"{1}.onnx",
                                                input_dict=testcases[1][1],
                                                index=1,
                                                output_import=False,
                                                opt_level=4,
                                                is_single_output=len(testcases[1][2].model.output_like.keys()) == 1)
                        output_str += f"output_name_dict = {bug_report.other_info['output_map']}\n"

                        output_str += "\n"
                        output_str += diff_two_tensors("output_0", "output_1", "output_name_dict", "tvm_opt_4")
                        output_str += "\n"
                        output_str += onnx_to_tvm(index=0,
                                                output_import=False,
                                                opt_level=0,
                                                is_single_output=len(testcases[0][2].model.output_like.keys()) == 1)
                        output_str += "\n"
                        output_str += onnx_to_tvm(index=1,
                                                output_import=False,
                                                opt_level=0,
                                                is_single_output=len(testcases[1][2].model.output_like.keys()) == 1)
                        output_str += "\n"
                        output_str += diff_two_tensors("output_0", "output_1", "output_name_dict", "tvm_opt_0")
                        output_str += "\n"

                    with open(os.path.join(output_path, f"test.py"), "w+") as f:
                        f.write(output_str)
                case "torch":
                    for i, testcase in enumerate(testcases):
                        output_str = onnx_to_torch(testcase[0], testcase[1], os.path.join(output_path, f"{i}.pickle"))
                        with open(os.path.join(output_path, f"test_{i}.py"), "w+") as f:
                            f.write(output_str)
                case "torchcompile":
                    for i, testcase in enumerate(testcases):
                        output_str = onnx_to_torch(testcase[0], testcase[1], os.path.join(output_path, f"{i}.pickle"),
                                                   use_pt_compile=True)
                        with open(os.path.join(output_path, f"test_{i}.py"), "w+") as f:
                            f.write(output_str)
                case "onnxruntime":
                    output_str = f"# input dict: " \
                                 f"{bug_report.testcase.model.corresponding_input_map(bug_report.eq_testcase.model)}\n"
                    output_str += "\n"

                    assert len(testcases) == 2

                    output_str += onnx_to_torch_code(model=testcases[0][2].model, 
                                                        input_dict=testcases[0][1],
                                                        data_output_path=os.path.join(output_path, f"{0}.pickle"), 
                                                        index=0, 
                                                        output_import=True,
                                                        device=backend_target)
                    output_str += "\n"
                    output_str += onnx_to_torch_code(model=testcases[1][2].model, 
                                                        input_dict=testcases[1][1],
                                                        data_output_path=os.path.join(output_path, f"{1}.pickle"), 
                                                        index=1, 
                                                        output_import=False)
                    output_str += "\n"
                    output_str += onnx_to_onnxruntime(
                        output_names=list(testcases[0][2].model.output_like.keys()),
                        onnx_path=f"{0}.onnx",
                        input_dict=testcases[0][1],
                        index=0,
                        output_import=False,
                        enable_opt=True,
                    )
                    output_str += "\n"
                    output_str += onnx_to_onnxruntime(
                        output_names=list(testcases[1][2].model.output_like.keys()),
                        onnx_path=f"{1}.onnx",
                        input_dict=testcases[1][1],
                        index=1,
                        output_import=False,
                        enable_opt=True
                    )
                    output_str += f"output_name_dict = {bug_report.other_info['output_map']}\n"

                    output_str += "\n"
                    output_str += diff_two_tensors("output_0", "output_1", "output_name_dict",
                                                   "onnxruntime_enable_opt")
                    output_str += "\n"
                    output_str += onnx_to_onnxruntime(
                        output_names=list(testcases[0][2].model.output_like.keys()),
                        onnx_path=f"{0}.onnx",
                        index=0,
                        output_import=False,
                        enable_opt=False
                    )
                    output_str += "\n"
                    output_str += onnx_to_onnxruntime(
                        output_names=list(testcases[1][2].model.output_like.keys()),
                        onnx_path=f"{1}.onnx",
                        index=1,
                        output_import=False,
                        enable_opt=False
                    )
                    output_str += "\n"
                    output_str += diff_two_tensors("output_0", "output_1", "output_name_dict",
                                                   "onnxruntime_disable_opt")
                    output_str += "\n"

                    output_str += onnx_to_tvm(onnx_path=f"{0}.onnx",
                                              index=0,
                                              output_import=False,
                                              opt_level=0,
                                              is_single_output=len(testcases[0][2].model.output_like.keys()) == 1)
                    output_str += "\n"
                    output_str += onnx_to_tvm(onnx_path=f"{1}.onnx",
                                              index=1,
                                              output_import=False,
                                              opt_level=0,
                                              is_single_output=len(testcases[1][2].model.output_like.keys()) == 1)
                    output_str += f"output_name_dict = {bug_report.other_info['output_map']}\n"

                    output_str += "\n"
                    output_str += diff_two_tensors("output_0", "output_1", "output_name_dict", "tvm_opt_4")
                    output_str += "\n"

                    with open(os.path.join(output_path, f"test.py"), "w+") as f:
                        f.write(output_str)
                case "onnxcompile":
                    assert isinstance(bug_report.testcase.model, ONNXModel)
                    input_dict = bug_report.testcase.oracle.input
                    output_str = onnx_to_torch_code(model=bug_report.testcase.model, 
                                                       input_dict=input_dict,
                                                       data_output_path=os.path.join(output_path, f"{0}.pickle"), 
                                                       index=0, 
                                                       output_import=True,
                                                       device=backend_target)
                    output_str += "\n"

                    if bug_report.eq_testcase is not None:
                        assert isinstance(bug_report.eq_testcase.model, ONNXModel)
                        input_dict = bug_report.eq_testcase.oracle.input
                        output_str += onnx_to_torch_code(model=bug_report.eq_testcase.model, 
                                                            input_dict=input_dict,
                                                            data_output_path=os.path.join(output_path, f"{1}.pickle"), 
                                                            index=1, 
                                                            output_import=False)

                    with open(os.path.join(output_path, f"test_complie.py"), "w+") as f:
                        f.write(output_str)
                case "tensorrt":
                    assert len(testcases) == 2
                    
                    output_str = generation_onnx_to_tensorrt(testcases[0][2].model, testcases[1][2].model, testcases[0][1], testcases[1][1], bug_report.other_info['output_map'], backend_target=backend_target)

                    with open(os.path.join(output_path, f"test.py"), "w+") as f:
                        f.write(output_str)

                case "infinitensor":
                    if len(testcases) == 2:
                        output_str = generation_onnx_to_infinitensor(testcases[0][2].model, testcases[1][2].model, testcases[0][1], testcases[1][1], bug_report.other_info['output_map'], backend_target=backend_target)
                    else:
                        output_str = ""
                        output_str += "\n".join([
                                "import numpy as np",
                                "from numpy import testing",
                                "import pickle",
                                "import onnx",
                                "import torch",
                                "import onnxruntime as ort",
                                "from pyinfinitensor.onnx import OnnxStub, backend"]
                            )
                        output_str += "\n\n"
                        output_str += onnx_to_torch_code(model=testcases[0][2].model, 
                                                         input_dict=testcases[0][1],
                                                         data_output_path=f"{0}.pickle", 
                                                         index=0, 
                                                         output_import=False,
                                                         device=backend_target
                                                        )
                        output_str += "\n"
                        output_str += onnx_to_infinitensor(onnx_path=f"{0}.onnx",
                                                     index=0,
                                                     device=backend_target)
                        output_str += "\n"
                    with open(os.path.join(output_path, f"test.py"), "w+") as f:
                        f.write(output_str)
                case _:
                    raise NotImplementedError
        case "torch":
            if bug_report.eq_testcase is not None:
                torch_path_1 = os.path.join(input_path, "t1", "model.pth")
                torch_path_2 = os.path.join(input_path, "t2", "model.pth")
                input_1 = bug_report.testcase.oracle.input
                input_2 = bug_report.eq_testcase.oracle.input
                # print(input_1)
                # print(input_2)
                testcases = [(torch_path_1, input_1, bug_report.testcase),
                             (torch_path_2, input_2, bug_report.eq_testcase)]
            else:  # deprecated
                torch_path = os.path.join(input_path, "model.pth")
                input_dict = bug_report.testcase.oracle.input
                testcases = [(torch_path, input_dict, bug_report.testcase)]

            match desired_code:
                case "torchcompile":
                    assert isinstance(bug_report.testcase.model, TorchModel)
                    input_dict = bug_report.testcase.oracle.input
                    render = torch_to_torch_compile(bug_report.testcase.model, input_dict,
                                                        os.path.join(output_path, "0.pickle"))
                    with open(os.path.join(output_path, f"test_0.py"), "w+") as f:
                        f.write(render.render())

                    if bug_report.eq_testcase is not None:
                        assert isinstance(bug_report.eq_testcase.model, TorchModel)
                        input_dict = bug_report.eq_testcase.oracle.input
                        output_str = torch_to_torch_compile(bug_report.eq_testcase.model, input_dict,
                                                            os.path.join(output_path, "1.pickle"))
                        with open(os.path.join(output_path, f"test_1.py"), "w+") as f:
                            f.write(output_str)
                case "torch":
                    assert len(testcases) <= 2

                    if len(testcases) == 1:
                        input_dict = testcases[0][1]
                        render = torch_to_torch_compile(testcases[0][2].model, input_dict,
                                                        os.path.join(output_path, "0.pickle"))
                        with open(os.path.join(output_path, f"test_0.py"), "w+") as f:
                            f.write(render.render())
                    else:
                    
                        output_str = generation_torch_to_torch_compile(testcases[0][2].model, testcases[1][2].model, testcases[0][1], testcases[1][1], bug_report.other_info['output_map'], backend_target)

                        with open(os.path.join(output_path, f"test.py"), "w+") as f:
                            f.write(output_str)

                case "hidet":
                    assert len(testcases) == 2
                    
                    output_str = generation_torch_to_hidet(testcases[0][2].model, testcases[1][2].model, testcases[0][1], testcases[1][1], bug_report.other_info['output_map'], backend_target)

                    with open(os.path.join(output_path, f"test.py"), "w+") as f:
                        f.write(output_str)
                case _:
                    raise NotImplementedError
        case "tensorflow":
            if bug_report.eq_testcase is not None:
                tf_path_1 = os.path.join(input_path, "t1", "model", "tfnet")
                tf_path_2 = os.path.join(input_path, "t2", "model", "tfnet")
                input_1 = bug_report.testcase.oracle.input
                input_2 = bug_report.eq_testcase.oracle.input
                testcases = [(tf_path_1, input_1, bug_report.testcase),
                             (tf_path_2, input_2, bug_report.eq_testcase)]
            else:  # deprecated
                tf_path = os.path.join(input_path, "model", "tfnet")
                input_dict = bug_report.testcase.oracle.input
                testcases = [(tf_path, input_dict)]
            match desired_code:
                case "xla":
                    assert len(testcases) == 2

                    output_str = generation_tensorflow_to_xla(testcases[0][0], testcases[1][0], testcases[0][1], testcases[1][1], bug_report.other_info['output_map'], backend_target)

                    with open(os.path.join(output_path, f"test.py"), "w+") as f:
                        f.write(output_str)
                case _:
                    raise NotImplementedError
        case _:
            raise NotImplementedError


if __name__ == '__main__':
    main()
