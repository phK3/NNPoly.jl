
import time
import sys
import glob

from dnnv.nn import parse
from dnnv.nn.transformers.simplifiers import simplify

from pathlib import Path


def main():
    assert len(sys.argv) >= 2, "expected 1 argument: input directory"

    onnx_dir = sys.argv[1]
    if len(sys.argv) == 3:
    	out_suffix = sys.argv[2]
    else:
    	out_suffix = 'simple'
    	
    for onnx_file in glob.glob(onnx_dir + "/*.onnx"):
        op_graph = parse(Path(onnx_file))

        print("Start simplification ...")
        t = time.perf_counter()
        simplified_graph = simplify(op_graph)
        t2 = time.perf_counter()

        print(f"simplifying time: {t2 - t}")

        print("exporting simplified graph ...")
        t = time.perf_counter()
        out_file = onnx_file.split('.')[0] + '_' + out_suffix + '.onnx'
        simplified_graph.export_onnx(out_file)
        t2 = time.perf_counter()
        print(f"exporting time: {t2 - t}")


if __name__ == "__main__":
    main()

