module CFRibbage


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


"Do one core CFR loop."
function CFR!(playbuffer1::Vector{Int64}, playbuffer2::Vector{Int64},
             showbuffer1::Vector{Int64}, showbuffer2::Vector{Int64},
             regretbuffer1::Vector{Int64}, regretbuffer2::Vector{Int64},
             rows1, rows2, h1::hType, h2::hType, sp1, sp2, di1::Int64, di2::Int64, turncard,
             known1::HType, known2::HType, IM1::InformationModel_Base, IM2::InformationModel_Base, db::DB)

    getPlayScores!(playbuffer1, playbuffer2, rows1, rows2, di1, di2, known1, known2, turncard.rank, IM1, IM2, db)
    getShowScores!(showbuffer1, showbuffer2, rows1, rows2, h1, sp1, h2, sp2, di1, di2, turncard)



    ## I'm not sure i need the regret buffers; i might be able to just update df right in the loop where i'm currently
    ## filling out regrets. I think it's too late for me to start working on this. So future me: read this note!


    n1 = nrow(rows1)
    n2 = nrow(rows2)

    for ix in 1:n1
        regretbuffer1[ix] = playbuffer1[di1] + showbuffer1[di1] - playbuffer1[ix] - showbuffer1[ix]
    end
    for ix in 1:n2
        regretbuffer2[ix] = -playbuffer2[di2] - showbuffer2[di2] + playbuffer2[ix] + showbuffer2[ix]
    end

    println(regretbuffer1[1:n1])
    println(regretbuffer2[1:n2])


end



function testCFR()

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


    pb1 = fill(0, 20)
    pb2 = fill(0, 20)
    sb1 = fill(0, 20)
    sb2 = fill(0, 20)
    rb1 = fill(0, 20)
    rb2 = fill(0, 20)

    IM1 = InformationModel_Base(db)
    IM2 = InformationModel_Base(db)


    CFR!(pb1, pb2, sb1, sb2, rb1, rb2, rows1, rows2, h1, h2, sp1, sp2, di1, di2, turncard, known1, known2, IM1, IM2, db)


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

    getPlayScores!(sb1, sb2, rows1, rows2, di1, di2, known1, known2, turncard.rank, IM1, IM2, db)


end


function testDiscard()
    (hand1, hand2, turncard, h1, h2, sp1, sp2) = dealHands(deck)

    rows1 = @view df[hRows[h1],:]
    rows2 = @view df[hRows[h2],:]

    di1 = getDiscard(rows1, 1)
    di2 = getDiscard(rows2, 2)

    return (di1, di2)
end


testCFR()



end # module CFRibbage
