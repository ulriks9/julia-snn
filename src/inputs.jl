include(string(@__DIR__)[1:24] * "\\dependencies.jl")
#generates a signal that can be processed by the network
#uses WAV and DSP to convert to signal
#25ms Hanning window
function gen_inputs(path::String)
    id = wavread(path)
    i = id[1]
    i = power(spectrogram(i[:,1], round(Int, 25e-3 * id[2]), window=hanning))
    i = scale(i, 0, 103000)
    i
end
#scales array between specified range
function scale(arr, a, b)
    max = maximum(arr)
    min = minimum(arr)

    for i = 1:length(arr)
        arr[i] = ((b - a) * (arr[i] - min) / (max - min)) + a
    end
    arr
end
