using Pkg
pkg"registry add https://github.com/MurrellGroup/MurrellGroupRegistry"
Pkg.activate(".")

Pkg.add(["JLD2", "Flux"])
Pkg.add(["CUDA", "cuDNN"]) #<- If GPU
Pkg.develop(path="../")

using ProFlowDemo, Flux, JLD2, Flowfusion

using CUDA #<- If GPU
device = gpu #<- If GPU
#device = identity #<- if no GPU

#Only need to run once:
#run(`wget https://huggingface.co/MurrellLab/ProFlowDemo/resolve/main/ProFlowDemo_chkpt_3.jld2`)

model_state = JLD2.load("ProFlowDemo_chkpt_3.jld2", "model_state")
loadedmodel = FlowcoderSC(384, 6, 6)
Flux.loadmodel!(loadedmodel, model_state)
testmode!(loadedmodel)
model = loadedmodel |> device

for len in 100:100:500
    b = dummy_batch(vcat(ones(Int,len), 2ones(Int,len), 3ones(Int,len)))
    g = flow_quickgen(b, model, steps = 100, d = device)
    export_pdb("test_$(len).pdb", g, b.chainids, b.resinds)
    prot = gen2prot(g, b.chainids, b.resinds)
    for i in 1:length(prot)
        println(">seq$i")
        println(prot[i].sequence)
    end
end