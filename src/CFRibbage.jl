module CFRibbage

export testPlay

include("base.jl")
include("analytics.jl")
include("training.jl")

using BenchmarkTools, ProfileView
using Random


deck = [Card(r,s) for r in 7:11 for s in 1:4]
db = initDB(deck)

const imask = generateIncludeMask(db)
const emask = generateExcludeMask(db)



function dealHands(deck::Vector{Card})
    shuffle!(deck)

    hand1 = deck[1:6]
    hand2 = deck[7:12]
    turncard = deck[13]

    (h1, sp1) = canonicalize(hand1)
    (h2, sp2) = canonicalize(hand2)

    return (hand1, hand2, turncard, h1, h2, sp1, sp2)
end





function testPlay()



    (hand1, hand2, turncard, h1, h2, sp1, sp2) = dealHands(deck)

    rows1 = @view db.df[db.hRows[h1], :]
    rows2 = @view db.df[db.hRows[h2], :]

    di1 = getDiscard(rows1, 1)
    di2 = getDiscard(rows2, 2)

    D1 = HType()
    for suit in rows1[di1,:].discard
        for r in suit
            D1[r] += 1
        end
    end
    H1 = counter(rows1[di1,:].playhand)

    D2 = HType()
    for suit in rows2[di2,:].discard
        for r in suit
            D2[r] += 1
        end
    end
    H2 = counter(rows2[di2,:].playhand)

    known1 = merge(H1, D1)
    known2 = merge(H2, D2)
    known1[turncard.rank] += 1
    known2[turncard.rank] += 1

    # println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    # println("p1 was dealt: ", hand1, "   and p2 was dealt: ", hand2)
    # println("the turn was ", turncard)
    # println("p1 knows: ", c2v(known1), "   and p2 knows: ", c2v(known2))



    IM1 = InformationModel_Base(db)
    IM2 = InformationModel_Base(db)

    sb1 = fill(0, nrow(rows1))
    sb2 = fill(0, nrow(rows2))

    getPlayResults!(sb1, sb2, rows1, rows2, di1, di2, known1, known2, turncard.rank, IM1, IM2, db)


end


function testDiscard()
    (hand1, hand2, turncard, h1, h2, sp1, sp2) = dealHands(deck)

    rows1 = @view df[hRows[h1],:]
    rows2 = @view df[hRows[h2],:]

    di1 = getDiscard(rows1, 1)
    di2 = getDiscard(rows2, 2)

    return (di1, di2)
end



testPlay()

ProfileView.@profview for i in 1:20 testPlay() end



end # module CFRibbage
