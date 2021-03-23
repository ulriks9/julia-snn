include("dependencies.jl")
include("network.jl")
include("inputs.jl")
include("params.jl")

global input_res = length(get_mfcc("media\\training\\GTZAN\\rock\\rock.00000.wav")[1,:])

function get_params()
    Params(#= dt =# 1 / 10, #= tau =# 4, #= v_t =# 20, #= v_0 =# -70, #= v =# -55,
        #= ref =# 0.00, #= cl =# 10, #= hid =# 20, #= win =# 25, #= a_plus =# 10, #= a_minus =# 5,
        #= tau_plus =# 5, #= tau_minus =# 5, #= in_w =# 1)
end
#runs training on the network for specified amount of samples
function run_training(n::Int64)
    classes = readdir("media\\training\\GTZAN")
    processed = String[]
    spikes = Array[]

    for i = 1 : n
        class = classes[round(Int, rand() * length(classes))]
        samples = readdir("media\\training\\GTZAN\\" * class)
        sample = "media\\training\\GTZAN\\" * class * "\\" * samples[round(Int, rand() * length(samples))]
        if !(sample in processed)
            push!(spikes, run_cycle(sample))
            push!(processed, sample)
        end
    end
    [processed, spikes]
end
#simulates voltage levels
function v_sim()
    layers = get_layers()
    synapses = load_arr("data\\synapses.jld")
    inputs = get_mfcc("media\\training\\GTZAN\\rock\\rock.00000.wav")
    p = get_params()

    println("cycle method time:")
    @time l = cycle(layers, inputs, synapses, p)

    v_levels = get_parray(l[1], 1, 5, 500)

    println("plot time:")
    @time plot(v_levels)
    xlabel!("Time")
    ylabel!("Membrane Potential (mV)")
end
#simulates spikes
function s_sim()
    layers = get_layers()
    synapses = load_arr("data\\synapses.jld")
    inputs = get_mfcc("media\\training\\GTZAN\\rock\\rock.00000.wav")
    p = get_params()

    @time l = cycle(layers, inputs, synapses, p)

    s = get_parray(l[2], 1, 5, 500)

    @time plot(s)
    xlabel!("Time")
    ylabel!("Membrane Potential (mV)")
end
#returns array of layers
function get_layers()
    p = get_params()
    [fill(p.v, input_res), fill(p.v, p.hid), fill(p.v, p.cl)]
end
#returns array of synapses with random floating point values in range 0:1
function get_synapses()
    p = get_params()

    layer_1 = Array{Array}(UndefInitializer(), p.hid)
    layer_2 = Array{Array}(UndefInitializer(), p.cl)

    for i = 1 : length(layer_1)
        layer_1[i] = rand(input_res)
        layer_1[i] = scale(layer_1[i], 0, 100)
    end
    for i = 1 : length(layer_2)
        layer_2[i] = rand(p.hid)
        layer_2[i] = scale(layer_2[i], 0, 100)
    end
    [layer_1, layer_2]
end
#converts array from network to something Plots.jl can work with
function get_parray(in, l, n, c)
    a = zeros(length(in))
    for i = 1 : length(in)
        a[i] = in[i][l][n]
    end
    resize!(a, c)
end
#runs one training cycle with specified input, returns # of times output neurons spiked
function run_cycle(path::AbstractString)
    p = get_params()
    layers = get_layers()
    inputs = get_mfcc(path)
    synapses = load_arr("data\\synapses.jld")
    l = cycle(layers, inputs, synapses, p)
    s = l[2]

    save_arr(l[3], "data\\synapses.jl")

    spikes = zeros(p.cl)

    for i = 1 : length(s)
        for j = 1 : length(s[i][3])
            if s[i][3][j] > 0
                spikes[j] += 1
            end
        end
    end
    spikes
end
