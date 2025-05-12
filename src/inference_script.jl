using Pkg
Pkg.activate(".")
Pkg.develop(path="../")

using ProFlowDemo, Flux, JLD2

model_state = JLD2.load("../models/ProFlowDemo_warmdown_checkpoint_3.jld2", "model_state")
cpumodel = FlowcoderSC(384, 6, 6)
Flux.loadmodel!(cpumodel, model_state)
testmode!(cpumodel)

b = dummy_batch(vcat(ones(Int,40), 2ones(Int,40), 3ones(Int,40)))
g = flow_quickgen(b, cpumodel, steps = 100)

export_pdb("test_11.pdb", g, b.chainids, b.resinds)

prot = gen2prot(g, b.chainids, b.resinds)

for i in 1:length(prot)
    println(">seq$i")
    println(prot[i].sequence)
end