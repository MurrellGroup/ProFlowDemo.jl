# ProFlowDemo

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/ProFlowDemo.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/ProFlowDemo.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/ProFlowDemo.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/ProFlowDemo.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/ProFlowDemo.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/ProFlowDemo.jl)

## Installation

```julia
using Pkg
pkg"registry add https://github.com/MurrellGroup/MurrellGroupRegistry"
Pkg.add(["JLD2", "Flux"])
Pkg.add(["CUDA", "cuDNN"]) #<- If GPU
pkg"add https://github.com/MurrellGroup/ProFlowDemo.jl"
```

## Gen example

```julia
using ProFlowDemo, Flux, JLD2

using CUDA #<- If GPU
device = gpu #<- If GPU
#device = identity #<- if no GPU

!("ProFlowDemo_chkpt_3.jld2" in readdir()) && run(`wget https://huggingface.co/MurrellLab/ProFlowDemo/resolve/main/ProFlowDemo_chkpt_3.jld2`)
model_state = JLD2.load("ProFlowDemo_chkpt_3.jld2", "model_state");
loadedmodel = FlowcoderSC(384, 6, 6);
Flux.loadmodel!(loadedmodel, model_state);
testmode!(loadedmodel);
model = loadedmodel |> device;


chain_lengths = [96,96,178, 178]
b = dummy_batch(chainids_from_lengths(chain_lengths))
g = flow_quickgen(b, model, steps = 500, d = device)
export_pdb("gens/"*join(string.(lengths),"_")*"-$(stps).pdb", g, b.chainids, b.resinds)
```
