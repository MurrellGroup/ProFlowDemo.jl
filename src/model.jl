struct CrossFrameIPAblock{A,B}
    ln::A
    ipa::B
end
Flux.@layer CrossFrameIPAblock
CrossFrameIPAblock(dim::Int, ipa; ln = Flux.AdaLN(dim, dim)) = CrossFrameIPAblock(ln, ipa)
function (ipa_block::CrossFrameIPAblock)(frames1::Rigid, frames2::Rigid, x; pair_feats = nothing, cond = nothing, mask = 0, kwargs...)
    T1 = (frames1.composed.inner.values, frames1.composed.outer.values)
    T2 = (frames2.composed.inner.values, frames2.composed.outer.values)
    lnx = Onion.lncall(ipa_block.ln, x, cond)
    x = x + ipa_block.ipa(T1, lnx, T2, lnx, zij = pair_feats, mask = mask, show_warnings = false, kwargs...) ./ 2
    return x
end

struct FlowcoderSC{L}
    layers::L
end
Flux.@layer FlowcoderSC
function FlowcoderSC(dim, depth, f_depth)
    layers = (;
        depth = depth,
        f_depth = f_depth,
        t_rff = RandomFourierFeatures(1 => dim, 1f0),
        cond_t_encoding = Dense(dim => dim, bias=false),
        AApre_t_encoding = Dense(dim => dim, bias=false),
        pair_rff = RandomFourierFeatures(2 => 64, 1f0),
        pair_project = Dense(64 => 32, bias=false),
        AAencoder = Dense(21 => dim, bias=false),
        selfcond_crossipa = [CrossFrameIPAblock(dim, IPA(IPA_settings(dim, c_z = 32)), ln = AdaLN(dim, dim)) for _ in 1:depth],
        selfcond_selfipa = [CrossFrameIPAblock(dim, IPA(IPA_settings(dim, c_z = 32)), ln = AdaLN(dim, dim)) for _ in 1:depth],
        ipa_blocks = [IPAblock(dim, IPA(IPA_settings(dim, c_z = 32)), ln1 = AdaLN(dim, dim), ln2 = AdaLN(dim, dim)) for _ in 1:depth],
        framemovers = [Framemover(dim) for _ in 1:f_depth],
        AAdecoder = Chain(StarGLU(dim, 3dim), Dense(dim => 21, bias=false)),
    )
    return FlowcoderSC(layers)
end
ipa(l, f, x, pf, c, m) = l(f, x, pair_feats = pf, cond = c, mask = m)
crossipa(l, f1, f2, x, pf, c, m) = l(f1, f2, x, pair_feats = pf, cond = c, mask = m)
function (fc::FlowcoderSC)(t, Xt, chainids, resinds; sc_frames = nothing)
    l = fc.layers
    pmask = Flux.Zygote.@ignore self_att_padding_mask(Xt[1].lmask)
    pre_z = Flux.Zygote.@ignore l.pair_rff(pair_encode(resinds, chainids))
    pair_feats = l.pair_project(pre_z)
    t_rff = Flux.Zygote.@ignore l.t_rff(t)
    cond = reshape(l.cond_t_encoding(t_rff), :, 1, size(t,2))
    frames = Translation(tensor(Xt[1])) ∘ Rotation(tensor(Xt[2]))
    AA_one_hots = tensor(Xt[3])
    x = l.AAencoder(AA_one_hots .+ 0)
    for i in 1:l.depth
        if sc_frames !== nothing
            x = Flux.Zygote.checkpointed(crossipa, l.selfcond_selfipa[i], sc_frames, sc_frames, x, pair_feats, cond, pmask)
            if mod(i, 2) == 0
                x = Flux.Zygote.checkpointed(crossipa, l.selfcond_crossipa[i], frames, sc_frames, x, pair_feats, cond, pmask)
            else
                x = Flux.Zygote.checkpointed(crossipa, l.selfcond_crossipa[i], sc_frames, frames, x, pair_feats, cond, pmask)
            end
        end
        x = Flux.Zygote.checkpointed(ipa, l.ipa_blocks[i], frames, x, pair_feats, cond, pmask)
        if i > l.depth - l.f_depth
            frames = l.framemovers[i - l.depth + l.f_depth](frames, x, t = t)
        end
    end
    aa_logits = l.AAdecoder(x .+ reshape(l.AApre_t_encoding(t_rff), :, 1, size(t,2)))   
    return frames, aa_logits
end