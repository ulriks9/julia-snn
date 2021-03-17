include("dependencies.jl")
#generates a signal that can be processed by the network
#uses WAV and DSP to convert to signal
#25ms Hanning window
function get_spectro(path::String)
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
#extracts MFCCs from specified WAV file
function get_mfcc(path::AbstractString)
    w = wavread(path)
    m = mfcc(w[1], w[2])
    scale(m[1], 0, 10000)
end
