using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path="../")

using ProFlowDemo, DLProteinFormats, InvariantPointAttention, Onion, Flux, CannotWaitForTheseOptimisers
using DLProteinFormats: PDBSimpleFlat500, batch_flatrecs, sample_batched_inds

dat = DLProteinFormats.load(DLProteinFormats.PDBSimpleFlat);

model = FlowcoderSC(384, 6, 6)
sched = burnin_learning_schedule(0.000005f0, 0.001f0, 1.05f0, 0.99995f0)
opt_state = Flux.setup(Muon(eta = sched.lr), model)

for epoch in 1:10
    batchinds = sample_batched_inds(dat)
    for (i, b) in enumerate(batchinds)
        bat = batch_flatrecs(dat[b])
        ts = training_sample(bat)
        sc_frames = nothing
        if rand() < 0.5
            sc_frames, _ = model(ts.t, ts.Xt, ts.chainids, ts.resinds)
        end
        l, grad = Flux.withgradient(model) do m
            fr, aalogs = m(ts.t, ts.Xt, ts.chainids, ts.resinds, sc_frames = sc_frames)
            l_loc, l_rot, l_aas = losses(fr, aalogs, ts)
            l_loc + l_rot + l_aas
        end
        Flux.update!(opt_state, model, grad[1])
        if mod(i, 10) == 0
            Flux.adjust!(opt_state, next_rate(sched))
        end
        println(l)
    end
    jldsave("model_epoch_$epoch.jld", model_state = Flux.state(cpu(model)), opt_state=cpu(opt_state))
end