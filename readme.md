## About PolyJuice

PolyJuice is a fuzzer that can generate equivalent computation graphs to test the correctness of tensor compilers.
Its basic idea is to generate an equivalent tensor programs for a given tensor program, and then compare the output of
the two programs to check the correctness of the compiler under test.

## Usage

Before running the tool, we should first set some environment variables to enable the quality saturation engine:

```bash
export PYTHONPATH="$PWD"
export EQ_HELPER_PATH="$PWD/libdl_compiler_fuzzer_helper.so"
```

Next, ``python3 nnsmith/cli/equivalent_fuzz.py --help`` shows how it works:

```text
== Config ==
Override anything in the config (foo.bar=value)

model:
  type: null
  path: ???
mgen:
  max_nodes: 5
  timeout_ms: 10000
  vulops: false
  method: symbolic-cinit
  save: nnsmith_output
  seed: null
  max_elem_per_tensor: 65536
  rank_choices: null
  dtype_choices: null
  include: null
  exclude: null
  patch_requires: []
  grad_check: false
backend:
  type: null
  optmax: true
  target: cpu
ad:
  type: null
cache:
  topset: true
debug:
  viz: false
  viz_fmt: png
fuzz:
  time: 14400
  root: ???
  seed: null
  crash_safe: false
  test_timeout: null
  save_test: null
  random_num: 1
filter:
  type: []
  patch: []
cmp:
  equal_nan: true
  raw_input: null
  oracle: auto
  with:
    type: null
    optmax: true
    target: cpu
  seed: null
  bug_presence: report
  save: null


Powered by Hydra (https://hydra.cc)
Use --hydra-help to view Hydra specific help
```

Basically, the usage is almost identical with NNSmith. We provide some examples to show how to test Inductor, XLA and
TVM with PolyJuice.

### Test Inductor

To test PyTorch Inductor, make sure you have installed PyTorch (https://pytorch.org/get-started/locally/). We can run
the following command to test Inductor:

```bash
python3 nnsmith/cli/equivalent_fuzz.py fuzz.time=10s model.type=torch backend.type=pt2 backend.target=cpu fuzz.root=/tmp/fuzz_report debug.viz=true mgen.max_nodes=5
```

It will test Inductor for 10 seconds and output bugs to the directory ``/tmp/fuzz_report`` if any.

### Test XLA

To test XLA, make sure you have installed TensorFlow (https://www.tensorflow.org/install). We can run the following
command to test XLA:

```bash
python3 nnsmith/cli/equivalent_fuzz.py fuzz.time=10s model.type=tensorflow backend.type=xla backend.target=cpu fuzz.root=/tmp/fuzz_report debug.viz=true mgen.max_nodes=5
```

It will test XLA for 10 seconds and output bugs to the directory ``/tmp/fuzz_report`` if any.

### Test TVM

To test TVM, make sure you have installed TVM (https://tvm.apache.org/docs/install/index.html). We can run the following
command to test TVM:

```bash
python3 nnsmith/cli/equivalent_fuzz.py fuzz.time=10s model.type=onnx backend.type=tvm backend.target=cpu fuzz.root=/tmp/fuzz_report debug.viz=true mgen.max_nodes=5
```

It will test TVM for 10 seconds and output bugs to the directory ``/tmp/fuzz_report`` if any.

## Add New Backend

To add a new backend, we need to implement the backend in the ``nnsmith/backend`` directory. There are some examples for
adapting a new backend. You can refer to ``pt2.py`` and ``hidet.py``.

## Acknowledgement

PolyJuice is built on the top of NNSmith, and relies on NNSmith's model generation to generate the initial tensor
program. In addition, PolyJuice reuse equality saturation engine from egg. We would like to thank the authors of NNSmith
and egg for their great work.