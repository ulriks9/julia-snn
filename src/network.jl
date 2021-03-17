include("dependencies.jl")
include("synapse.jl")
include("neuron.jl")
include("params.jl")

#sets neurons spiked to reset
function l_reset(l, p::Params)
    for i = 1 : length(l)
        for j = length(l[i])
            if l[i][j] >= p.v_t
                l[i][j] = p.v_0
            end
        end
    end
    l
end

function cycle(layers, inputs, synapses, p::Params)
    #simulation time, set to the amount of timesteps of the song
    t_s = length(inputs[:,1])
    #input resolution of environment, set to the amount of frequencies of the song
    input_res = length(inputs[1,:])
    #array of spike times
    spikes = Array{Array}(UndefInitializer(), convert(Int, t_s / p.dt))
    #monitoring of voltage levels
    v_array = Array{Array}(UndefInitializer(), convert(Int, t_s / p.dt))
    #initializes arrays with correct structure
    for x = 1 : length(spikes)
        spikes[x] = [zeros(input_res), zeros(p.hid), zeros(p.cl)]
        v_array[x] = [zeros(input_res), zeros(p.hid), zeros(p.cl)]
    end
    #refractory periods
    ref = [zeros(input_res), zeros(p.hid), zeros(p.cl)]
    #index
    i = 1
    #time set to 0
    t = 1
    #position
    pos = 1
    #set starting voltage
    v = p.v

    while t < t_s - p.dt
        layers = l_reset(layers, p)

        for m = 1 : length(layers)
            for n = 1 : length(layers[m])
                #checks if neuron is recovering from a spike
                if ref[m][n] > 0
                    v_array[i][m][n] = p.v_0
                    layers[m][n] = p.v_0
                    spikes[i][m][n] = 0
                    ref[m][n] -= 1
                    continue
                end
                #different step for first layer
                if m == 1
                    #computes new voltage level
                    v = layers[m][n] + compute(inputs[pos,n], layers[m][n], synapses[m][n], p)
                else
                    #sums outputs from previous layer
                    for o = 1 : length(layers[m-1])
                        #v += p.s * compute(spikes[o,m-1,i], layers[m][n], synapses[m][n], p)
                        vt = compute(spikes[i][m-1][o], layers[m][n], synapses[m][n], p)
                        #net activation from input neurons
                        layers[m][n] += vt
                    end
                    v = layers[m][n]
                end
                #if voltage is above threshold, produce spike
                if v >= p.v_t
                    spikes[i][m][n] = p.v_t
                    layers[m][n] = p.v_t
                    v_array[i][m][n] = p.v_t
                    ref[m][n] = 1
                else
                    v_array[i][m][n] = v
                    layers[m][n] = v
                    spikes[i][m][n] = 0
                end
            end
        end
        t += p.dt
        i += 1

        if t % 1 == 0
            pos += 1
        end
    end
    [v_array, spikes]
end
