module CFRibbage

include("base.jl")


deck = standardDeck[1:20]

(df, hRows, HRows, allh, allH, HID, Hprobs_dealer, Hprobs_pone) = buildDB(deck)
const M = buildM(allH, HID)
saveM()



end # module CFRibbage
