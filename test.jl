include("network.jl")

spike_train = rand(Bool, 3,5)


nn = Network(;input_size=3, output_size=2, hidden_neurons=3, runtime=5)
nn.construct(nn; n₁=1, n₂=3)
nn.construct(nn; n₁=1, n₂=2)

nn.activate(nn, spike_train; θ=0.1, r=1, η=0.1, γᵥ=0.3, γₚ=0.3, γₘ=0.3, A₊=2.0, A₋=1.0, w₊=1.5, w₋=-0.8)