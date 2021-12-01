mutable struct Network
    num_neurons::Int64
    runtime::Int64
    threshold::Float32
    refractory::Int64

    weights::Array{Float32}
    voltage::Array{Float32}
    output::Array{Float32}

    fire_rec::Array{Float32}

    Pᵢ::Array{Float32}
    M::Array{Float32}
    
    construct
    activate

    function Network(;num_neurons::Int64, runtime::Int64, threshold::Float32, refractory::Int64)
        weights = zeros(Float32, num_neurons, num_neurons)
        voltage = zeros(Float32, num_neurons, 1)
        output = zeros(Bool, num_neurons, 1)
        fire_rec = fill(-refractory, num_neurons, 1)

        Pᵢ = zeros(Float32, num_neurons, 1)
        M = zeros(Float32, num_neurons, 1)

        new(num_neurons, runtime, threshold, refractory, weights, voltage, output, fire_rec, Pᵢ, M, construct, activate)
    end

    function construct()
        
    end

    function activate(self::Network)
        for t in 1:self.runtime
            self.voltage .= self.weights * self.output + self.voltage
            self.output .=  (self.voltage .>= self.threshold) .* ((t .- self.fire_rec) .> self.refractory)
        end
    end
end