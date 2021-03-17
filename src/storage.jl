include("dependencies.jl")
#saves array to CSV file
function save_csv(in, path::AbstractString)
    #stores length of array as it's used multiple times
    l = length(in)
    #creates DataFrame object to be saved
    out = DataFrame()
    #sets index 1 of out to the in array
    out[!, :A] = in
    #saves array to CSV file
    CSV.write(path, out)
end
#loads CSV file into an array
function load_csv(path)
    #reads CSV file specified
    d = CSV.read(path, DataFrame)
    #returns the column at index :A
    d[!, :A]
end
