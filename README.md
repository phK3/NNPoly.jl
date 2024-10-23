# NNPoly

A tool for neural network verification with optimizable polynomial overapproximations.

Have a look at the [demo_notebook.ipynb](https://github.com/phK3/NNPoly.jl/blob/main/demo_notebook.ipynb) for usage examples.

## Quickstart

Keep in mind that Julia uses a JIT compiler -- the first time you execute your code may be slow due to compilation overhead, subsequent runs are faster.

- Go to the home directory of this repository and type start Julia (simply type `julia` in your shell).
- type `]` to activate Julia's package manager
- type `activate .` to activate the environment of this directory

(or activate any other directory where you've installed `NNPoly.jl`)

Then, you can execute the following code to verify `vnnlib` properties.
```julia
using NNPoly
const NP = NNPoly  # just to have an abbreviation

# define the solver configuration you want to use
solver = NP.PolyCROWN(NP.DiffNNPolySym(common_generators=true), prune_neurons=true)

# verify the first 3 properties described in ./eval/mnist_fc/instances.csv
NP.verify_vnnlib(solver, "./eval/mnist_fc", max_properties=3, loss_fun=NP.violation_loss)
```

## Installation

An installation of Julia is required.

- clone the `NNPoly.jl` github repo
- navigate to the repo's root
- start Julia (simply type `julia` in your shell)
- type `]` to start the Julia package manager

Then execute
```julia
] activate .
] rm NeuralVerification, VnnlibParser, OnnxReader
] add https://github.com/phk3/VnnlibParser.jl
] add https://github.com/phk3/OnnxReader.jl
] add https://github.com/phk3/NeuralVerification.jl
] instantiate
```

The above packages have to be removed and then reinstalled as they are not available via Julia's general registry and are thus not identifiable only by their name.