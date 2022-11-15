module CFRibbage

include("base.jl")
include("analytics.jl")
include("training.jl")

using BenchmarkTools, Random


deck = [Card(r,s) for r in 7:11 for s in 1:4]
(df, hRows, HRows, allh, allH, hID, HID, Hprobs_dealer, Hprobs_pone) = buildDB(deck)
const global n_H = length(allH)
const global n_h = length(allh)

const imask = generateIncludeMask()
const emask = generateExcludeMask()


M = buildM(allH, HID)



deal1 = [7, 7, 8, 9, 10, 10]
H1 = counter(deal1[1:4])
known_1 = counter(deal1)

deal2 = [8, 8, 8, 9, 11, 11]
H2 = counter(deal2[1:4])
known_2 = counter(deal2)

turn = 9
known_1[turn] += 1
known_2[turn] += 1

IM1 = InformationModel_Base()
init_model!(IM1, known_1, 1, H1, M, HID)
init_trees!(IM1, 1, H1, M, HID)
init_probs!(IM1, allH, Hprobs_pone)

IM2 = InformationModel_Base()
init_model!(IM2, known_2, 2, H2, M, HID)
init_trees!(IM2, 2, H2, M, HID)
init_probs!(IM2, allH, Hprobs_dealer)

@time playHand(H1, H2, IM1, IM2)

end # module CFRibbage
