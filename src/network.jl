include("dependencies.jl")
include("synapse.jl")
include("neuron.jl")
include("params.jl")
include("storage.jl")

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
    #useful for setting parameters
    stdp_counter = 0
    #simulation time, set to the amount of timesteps of the song
    t_s = length(inputs[:,1])
    #input resolution of environment, set to the amount of frequencies of the song
    input_res = length(inputs[1,:])
    #array of spike times
    spikes = Array{Array}(UndefInitializer(), convert(Int, t_s / p.dt))
    stdp_spikes = Array{Array}(UndefInitializer(), convert(Int, t_s / p.dt))
    #monitoring of voltage levels
    v_array = Array{Array}(UndefInitializer(), convert(Int, t_s / p.dt))
    #initializes arrays with correct structure
    for x = 1 : length(spikes)
        spikes[x] = [zeros(input_res), zeros(p.hid), zeros(p.cl)]
        stdp_spikes[x] = [zeros(input_res), zeros(p.hid), zeros(p.cl)]
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
    #main iterator, goes through the inputs one timestep at a time
    while t < t_s - p.dt
        layers = l_reset(layers, p)
        #iterates through layers
        for m = 1 : length(layers)
            #iterates through each neuron in layer
            for n = 1 : length(layers[m])
                #checks if neuron is recovering from a spike
                if ref[m][n] > 0
                    v_array[i][m][n] = p.v_0
                    layers[m][n] = p.v_0
                    spikes[i][m][n] = 0
                    stdp_spikes[i][m][n] = 0
                    ref[m][n] -= 1
                    continue
                end
                #different step for first layer
                if m == 1
                    #computes new voltage level
                    #instead of passing a synapse as parameter, it uses p.in_w for the input layer
                    v = layers[m][n] + compute(inputs[pos,n], layers[m][n], p.in_w, p)
                else
                    #sums outputs from previous layer
                    for o = 1 : length(layers[m-1])
                        vt = compute(spikes[i][m-1][o], layers[m][n], synapses[m-1][n][o], p)
                        #net activation from input neurons
                        layers[m][n] += vt
                    end
                    v = layers[m][n]
                end
                #if voltage is above threshold, produce spike
                if v >= p.v_t
                    spikes[i][m][n] = p.v_t
                    stdp_spikes[i][m][n] = p.v_t
                    layers[m][n] = p.v_t
                    v_array[i][m][n] = p.v_t
                    ref[m][n] = 1
                else
                    v_array[i][m][n] = v
                    layers[m][n] = v
                end
                #occurs every p.win timesteps
                #current implementation of STDP
                if i % p.win == 0 && i > 0
                    t_start = i - p.win + 1
                    t_end = i
                    t_curr = t_start
                    t_pre = t_start
                    t_post = t_start
                    #creates a sub array of spikes in the time window specified
                    temp_spikes = stdp_spikes[t_start:t_end]
                    #iterates through layers, starts at 2 as layer 1 has constant weights
                    for x = 2 : length(layers)
                        #iterates through neurons of that layer
                        for y = 1 : length(layers[x])
                            #iterates through timesteps in window
                            for z = 1 : p.win
                                #finds when neuron spiked
                                if temp_spikes[z][x][y] > 0
                                    t_post = t_curr
                                    #iterates through neurons of previous layer
                                    for a = 1 : length(layers[x-1])
                                        not_found = true
                                        count = 1
                                        #iterates through timesteps in window
                                        while not_found && count < p.win
                                            #finds when pre neuron spiked EITHER a or y
                                            if temp_spikes[count][x-1][a] > 0
                                                t_pre = t_start + count
                                                #updates weight for synapse to the post neuron
                                                synapses[x-1][y][a] += stdp(t_pre, t_post, p)
                                                stdp_counter += 1
                                                #makes sure weights aren't negative
                                                if synapses[x-1][y][a] < 0
                                                    synapses[x-1][y][a] = 0
                                                end
                                                #removes spike examined from the array of spikes to not analyse the same spike twice
                                                temp_spikes[count][x-1][a] = 0
                                                not_found = false
                                            end
                                            count += 1
                                        end
                                    end
                                    #removes spike examined from temporary array of spikes
                                    temp_spikes[z][x][y] = 0
                                end
                                t_curr += 1
                            end
                        end
                    end
                end
            end
        end
        t += p.dt
        i += 1
        #updates position at every whole number step
        if t % 1 == 0
            pos += 1
        end
    end
    println("STDP Performed " * string(stdp_counter) * " times")
    [v_array, spikes, synapses]
end
