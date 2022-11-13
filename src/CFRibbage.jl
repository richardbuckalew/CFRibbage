module CFRibbage

include("base.jl")
include("analytics.jl")

using Random

deck = [Card(r,s) for r in 5:11 for s in 1:2]

(df, hRows, HRows, allh, allH, hID, HID, Hprobs_dealer, Hprobs_pone) = buildDB(deck)

df.dealt_dealer = rand(0:10, nrow(df))


tData = TrainingData(1000, rand(1:100, 1000), rand(1:100, 1000), [], [], rand(1:100, 1000), rand(1:10, 1000))
old_stats = SummaryStats(0, 0.0, 0.0, Accumulator{Int64, Int64}(), Accumulator{Int64, Int64}(), 
                        Accumulator{Int64, Int64}(), Accumulator{Int64, Int64}(), 0.0, 0.0, 0.0, 
                        (Float64[], Int64[]), (Float64[], Int64[]))


S = summarize(df, hRows, Hprobs_dealer, Hprobs_pone, old_stats, tData)

display(S)


end # module CFRibbage
