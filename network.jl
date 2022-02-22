mutable struct Network
    input_size::Int64
    output_size::Int64
    hidden_neurons::Int64
    runtime::Int64

    weights::Array{Float32}
    voltage::Array{Float32}
    output::Array{Bool}

    fire_rec::Array{Int64}

    Pᵢ::Array{Float32}
    M::Array{Float32}
    
    construct
    activate

    function Network(;input_size::Int64, output_size::Int64, hidden_neurons::Int64, runtime::Int64)
        weights = rand(Float32, hidden_neurons+output_size, input_size+hidden_neurons)
        for i in 1:hidden_neurons
            weights[input_size+i, hidden_neurons] = 0.0f0
        end
        voltage = zeros(Float32, hidden_neurons+output_size, 1)
        output = zeros(Bool, input_size+hidden_neurons, 1)
        fire_rec = fill(-refractory, hidden_neurons+output_size, 1)

        Pᵢ = zeros(Float32, hidden_neurons+output_size, input_size+hidden_neurons)
        M = zeros(Float32, hidden_neurons+output_size, 1)

        new(input_size, output_size, hidden_neurons, runtime, weights, voltage, output, fire_rec, Pᵢ, M, construct, activate)
    end

    function construct(self::Network; n₁::Int64, n₂::Int64)
        self.weights[n₁+self.input_size, n₂] = (-1)^rand(Bool)*rand(Float32)
        self.weights[n₂+self.input_size, n₁] = 0.0f0
    end

    # θ : threshold, r : refractory
    function activate(self::Network, spike_train::Array{Bool}; θ, r, η, γᵥ, γₚ, γₘ, A₊, A₋, w₊, w₋)
        for t in 1:self.runtime
            # load input data
            self.output[1:self.input_size, 1] .= spike_train[:,t]

            # load result
            self.output[self.input_size+1:end, 1] .= self.voltage[1:self.hidden_neurons, 1] .>= θ

            # update the fire records
            self.fire_rec .= floor.(Int, self.fire_rec .* (t ./ self.fire_rec) .^ (self.voltage .>= θ))

            # reset voltage after firing
            self.voltage -= (self.voltage .>= θ) .* self.voltage

            # update pre-synaptic trace
            self.Pᵢ .*= γₚ
            for i in 1:self.hidden_neurons+self.output_size
                if t-self.fire_rec[i,1] > r 
                    self.Pᵢ[i,:] .+= self.output[:,1] .* (self.weights[i,:] .!= 0.0f0) * A₊
                    self.weights[i,:] .+= η * self.output[:,1] .* (self.weights[i,:] .!= 0.0f0) * self.M[i,1] * (w₋ .- self.weights[i,:])
                end
            end

            self.voltage .*= γᵥ
            self.voltage .= self.weights * self.output .* ((t .- self.fire_rec) .> r) + self.voltage
            self.output[self.input_size+1:end, 1] .= self.voltage[1:self.hidden_neurons, 1] .>= θ
            for i in axes(weights, 1) 
                self.weights[i,:] .+= η * self.output[i,1] .* (self.weights[i,:] .!= 0.0f0) .* self.Pᵢ[i,:] * (w₊ .- self.weights[i,:])
            end

            # update post-synaptic trace
            self.M .*= γₘ
            self.M .-= (self.voltage[1:self.hidden_neurons, 1] .>= θ) * A₋
        end
    end
end
