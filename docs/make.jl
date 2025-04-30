using ProFlowDemo
using Documenter

DocMeta.setdocmeta!(ProFlowDemo, :DocTestSetup, :(using ProFlowDemo); recursive=true)

makedocs(;
    modules=[ProFlowDemo],
    authors="murrellb <murrellb@gmail.com> and contributors",
    sitename="ProFlowDemo.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/ProFlowDemo.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/ProFlowDemo.jl",
    devbranch="main",
)
