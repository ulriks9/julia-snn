using WAV
using DSP
using LinearAlgebra
#generates a signal that can be processed by the network
#uses WAV and DSP to convert to signal
#25ms Hanning window
function gen_inputs(path::String)
    i = wavread(path)
    id = i[1]
    normalize(power(spectrogram(id[:,1], convert(Int, 25e-3 * i[2]), window=hanning)))
end
