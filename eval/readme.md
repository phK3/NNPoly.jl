
# Preparing Benchmarks for CORA

First simplify .onnx networks s.t. we only have linear and relu layers.

```
conda activate dnnv
python simplify_onnx_dir <path/to/onnx/networks/folder> <suffix> 
```

e.g. if suffix is 'simple' then we create files `<network>_simple.onnx` in the original directory.


Then create vnnlib properties that can be parsed (i.e. no complicated or/and constructions for the output).

```
conda activate dnnv
python rewrite_output_property.py <path/to/vnnlib/folder> <output_dir> <suffix>
```
