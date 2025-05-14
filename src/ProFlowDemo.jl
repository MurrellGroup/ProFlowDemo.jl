module ProFlowDemo

    using Flowfusion, ForwardBackward, Flux, RandomFeatureMaps, Onion, InvariantPointAttention, BatchedTransformations, ProteinChains, DLProteinFormats

    include("flow.jl")
    include("model.jl")

    chainids_from_lengths(lengths) = vcat([repeat([i],l) for (i,l) in enumerate(lengths)]...)
    gen2prot(samp, chainids, resnums) = DLProteinFormats.unflatten(tensor(samp[1]), tensor(samp[2]), tensor(samp[3]), clamp.(chainids, 0, 9), resnums)[1]
    export_pdb(path, samp, chainids, resnums) = ProteinChains.writepdb(path, gen2prot(samp, chainids, resnums))

    function dummy_batch(chainid_vec)
        L = length(chainid_vec)
        chainids = reshape(chainid_vec, :, 1)
        resinds = similar(chainids)[:] .= 1:L
        padmask = trues(L, 1)
        aas = 21*ones(Int, L, 1)
        locs = randn(Float32, 3, 1, L, 1)
        return (;chainids, resinds, padmask, aas, locs)
    end

    export training_sample, P, FlowcoderSC, losses, flow_quickgen, export_pdb, gen2prot, dummy_batch

end
