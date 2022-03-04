using Combinatorics, Serialization, IterTools, ProgressMeter, StatsBase, ProfileView, BenchmarkTools
using DataStructures, Random, DataFrames, ThreadTools, Infiltrator, LinearAlgebra, TimerOutputs


# (@isdefined to) || (const to = TimerOutput())

struct Card
    rank::Int64;
    suit::Int64;    
end
Base.show(io::IO, c::Card) = print(io, shortname(c));
showHand(H) = display(permutedims(H));
Base.isless(x::Card, y::Card) = (x.suit > y.suit) ? true : (x.rank < y.rank);   ## Bridge bidding suit order, ace == 1


function countertovector(c::Accumulator{Int64, Int64})
    return sort([r for r in keys(c) for ii in 1:c[r]]);
end



(@isdefined cardsuits) || (const cardsuits = collect(1:4));
(@isdefined cardranks) || (const cardranks = collect(1:13));
(@isdefined cardvalues) || (const cardvalues = vcat(collect(1:10), [10, 10, 10]));
(@isdefined suitnames) || (const suitnames = ["spades", "hearts", "diamonds", "clubs"]);
(@isdefined ranknames) || (const ranknames = ["ace", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "jack", "queen", "king"]);
(@isdefined shortsuit) || (const shortsuit = ["♠","♡","♢","♣"]);
(@isdefined shortrank) || (const shortrank = ["A","2","3","4","5","6","7","8","9","T","J","Q","K"]);

(@isdefined PP) || (const PP = [
    [[1]],
    [[2], [1,1]],
    [[3], [2,1], [1,2], [1,1,1]],
    [[4], [3,1], [1,3], [2,2], [2,1,1], [1,2,1], [1,1,2], [1,1,1,1]]
]);
(@isdefined cardcombs) || (const cardcombs = [collect(combinations(cardranks, n)) for n in 1:4]);


(@isdefined standardDeck) || (const standardDeck = [Card(r,s) for r in cardranks for s in cardsuits]);
(@isdefined rankDeck) || (const rankDeck = [c.rank for c in standardDeck]);

(@isdefined fullname) || (const fullname(c::Card) = ranknames[c.rank] * " of " * suitnames[c.suit]);
(@isdefined shortname) || (const shortname(c::Card) = shortrank[c.rank] * shortsuit[c.suit]);
(@isdefined handname) || (const handname(h::AbstractArray{Card}) = permutedims(shortname.(sort(h, by = c->c.rank))));

(@isdefined handranks) || (handranks(h::Vector{Card}) = sort([c.rank for c in h]));
(@isdefined handsuits) || (handsuits(h::Vector{Card}) = sort([c.suit for c in h]));


const hType = Accumulator{Int64, Int64}
const handType = Union{
    Tuple{Tuple{Int64, Int64, Int64, Int64, Int64, Int64}, Tuple{}, Tuple{}, Tuple{}},
    Tuple{Tuple{Int64, Int64, Int64, Int64, Int64}, Tuple{Int64}, Tuple{}, Tuple{}},
    Tuple{Tuple{Int64, Int64, Int64, Int64}, Tuple{Int64, Int64}, Tuple{}, Tuple{}},
    Tuple{Tuple{Int64, Int64, Int64, Int64}, Tuple{Int64}, Tuple{Int64}, Tuple{}},
    Tuple{Tuple{Int64, Int64, Int64}, Tuple{Int64, Int64, Int64}, Tuple{}, Tuple{}},
    Tuple{Tuple{Int64, Int64, Int64}, Tuple{Int64, Int64}, Tuple{Int64}, Tuple{}},
    Tuple{Tuple{Int64, Int64, Int64}, Tuple{Int64}, Tuple{Int64}, Tuple{Int64}},
    Tuple{Tuple{Int64, Int64}, Tuple{Int64, Int64}, Tuple{Int64, Int64}, Tuple{}},
    Tuple{Tuple{Int64, Int64}, Tuple{Int64, Int64}, Tuple{Int64}, Tuple{Int64}}
}
const discardType = Union{
    Tuple{Tuple{Int64, Int64}, Tuple{}, Tuple{}, Tuple{}},
    Tuple{Tuple{}, Tuple{Int64, Int64}, Tuple{}, Tuple{}},
    Tuple{Tuple{}, Tuple{}, Tuple{Int64, Int64}, Tuple{}},
    Tuple{Tuple{}, Tuple{}, Tuple{}, Tuple{Int64, Int64}},
    Tuple{Tuple{Int64}, Tuple{Int64}, Tuple{}, Tuple{}},
    Tuple{Tuple{Int64}, Tuple{}, Tuple{Int64}, Tuple{}},
    Tuple{Tuple{Int64}, Tuple{}, Tuple{}, Tuple{Int64}},
    Tuple{Tuple{}, Tuple{Int64}, Tuple{Int64}, Tuple{}},
    Tuple{Tuple{}, Tuple{Int64}, Tuple{}, Tuple{Int64}},
    Tuple{Tuple{}, Tuple{}, Tuple{Int64}, Tuple{Int64}},
}


include("playUtils.jl");
include("dbUtils.jl");



(@isdefined db) || (@time const db = deserialize("db.jls"))
(@isdefined phDealerProbs) || (@time const phDealerProbs = deserialize("phDealerProbs.jls"))
(@isdefined phPoneProbs) || (@time const phPoneProbs = deserialize("phPoneProbs.jls"))
(@isdefined handID) || (@time const handID = deserialize("handID.jls"))  # Tuple => NTuple{2, Int64}
(@isdefined allPH) || (@time const allPH = deserialize("allPH.jls"))     # Vector{Counter}
(@isdefined phID) || (@time const phID = deserialize("phID.jls"))        # Counter => Int64
(@isdefined phRows) || (@time const phRows = deserialize("phRows.jls"))  # Counter => Vector{Int64}
(@isdefined M) || (@time global M = loadM())




function canonicalize(h::Vector{Card})
    H = [[c.rank for c in h if c.suit == ii] for ii in 1:4]
    sort!.(H)  # sort each suit
    sp1 = sortperm(H)   # sort suits lexicographically
    H = H[sp1]
    sp2 = sortperm(H, by = length, rev = true)  # subsort by length, longest to shortest
    H = H[sp2]
    sp = sp1[sp2] # suit permutation (for reconstructing h)
    return (Tuple(Tuple.(H)), sp)
end

function unCanonicalize(h, sp)
    hand = Card[]
    for (ii, s) in enumerate(sp)
        for r in h[ii]
            (r == 0) && continue
            push!(hand, Card(r, s))
        end
    end
    return hand
end

function scoreShow(h::Vector{Card}, turn::Card; isCrib = false)

    cCombs = collect.([combinations(vcat(h,[turn]), ii) for ii in 2:(length(h)+1)])

    ranks = sort(vcat(handranks(h), handranks([turn])))
    rCombs = [[handranks(c) for c in comb] for comb in cCombs]

    nPairs = count([x[1] == x[2] for x in rCombs[1]])
    pairScore = 2 * nPairs
    # display("pairs: " * string(pairScore))

    nRuns = 0
    runLength = 0
    for ell in (length(ranks)-1):-1:2
        runLength = ell + 1
        adjRanks = [r .- minimum(r) .+ 1 for r in rCombs[ell]]
        nRuns = count(map(isequal(collect(1:runLength)), adjRanks))
        (nRuns > 0) && break
    end
    runScore = nRuns * runLength
    # display("runs: " * string(runScore))
    
    vCombs = [[cardvalues[c] for c in comb] for comb in rCombs]
    S = sum.(vcat(vCombs...))
    nFifteens = count(S .== 15)
    fifteenScore = 2 * nFifteens
    # display("fifteens: " * string(fifteenScore))


    if isCrib
        suits = sort([c.suit for c in h])
    else
        suits = sort([c.suit for c in vcat(h, [turn])])
    end
    flushScore = 0
    for ell in length(suits):-1:4
        sCombs = collect(combinations(suits, ell))
        sLens = length.(Set.(sCombs))
        if any(sLens .== 1)
            flushScore = ell
            break
        end
    end
    # display("flush: " * string(flushScore))

    return pairScore + runScore + fifteenScore + flushScore


end

function dealHands(deck = standardDeck, handSize = 6)
    dealtCards = sample(deck, handSize * 2 + 1, replace = false)
    h1 = dealtCards[1:handSize]
    h2 = dealtCards[handSize+1:2*handSize]
    turn = dealtCards[end]
    return (h1, h2, turn)
end

function makeCrib(d1, d2)
    return vcat(d1, d2)
end

function stripsuits(D::discardType)::Vector{Int64}
    return vcat([[x...] for x in D]...);
end


function doCFR()

    (h1Cards, h2Cards, turnCard) = dealHands()

    (h1, sp1) = canonicalize(h1Cards)
    (h2, sp2) = canonicalize(h2Cards)

    hi1 = handID[h1]
    h1Rows = hi1[1]:hi1[2]
    hi2 = handID[h2]
    h2Rows = hi2[1]:hi2[2]

    allD1 = @view db[h1Rows, :discard]
    allD1Cards = [unCanonicalize(d, sp1) for d in allD1]
    allD1Ranks = [stripsuits(D) for D in allD1]
    allD2 = @view db[h2Rows, :discard]
    allD2Cards = [unCanonicalize(d, sp2) for d in allD2]
    allD2Ranks = [stripsuits(D) for D in allD2]
    
    p1PlayHands = @view db[h1Rows, :playhand]
    p2PlayHands = @view db[h2Rows, :playhand]

    p1weights = ProbabilityWeights(view(db, h1Rows, :dealerprofile))
    p2weights = ProbabilityWeights(view(db, h2Rows, :poneprofile))
    di1 = sample(1:length(allD1), p1weights)
    di2 = sample(1:length(allD2), p2weights)

    p1playhand = p1PlayHands[di1]
    d1h = allD1Ranks[di1]
    p2playhand = p2PlayHands[di2]
    d2h = allD2Ranks[di2]

    p1PlayMargins = [naiveplay([[p1playhand...], [p2h...]], [d1h, p2d], turnCard.rank)[1] for (p2h, p2d) in zip(p2PlayHands, allD2Ranks)]
    p2PlayMargins = [naiveplay([[p1h...], [p2playhand...]], [p1d, d2h], turnCard.rank)[1] for (p1h, p1d) in zip(p1PlayHands, allD1Ranks)]

    p1ShowHands = [setdiff(h1Cards, d) for d in allD1Cards]
    p2ShowHands = [setdiff(h2Cards, d) for d in allD2Cards]

    p1Cribs = [makeCrib(d1, allD2Cards[di2]) for d1 in allD1Cards]
    p2Cribs = [makeCrib(allD1Cards[di1], d2) for d2 in allD2Cards]

    p1ShowScores = [scoreShow(H, turnCard) for H in p1ShowHands]
    p1CribScores = [scoreShow(C, turnCard, isCrib = true) for C in p1Cribs]
    p2ShowScores = [scoreShow(H, turnCard) for H in p2ShowHands]
    p2CribScores = [scoreShow(C, turnCard, isCrib = true) for C in p2Cribs]
    p1ShowMargins = [ss - p2ShowScores[di2] for ss in p1ShowScores] .+ p1CribScores     # actual scores: maximize this
    p2ShowMargins = [p1ShowScores[di1] - ss for ss in p2ShowScores] .+ p2CribScores     # actual scores: minimize this

    p1Objectives = p1PlayMargins .+ p1ShowMargins
    p2Objectives = -p2PlayMargins .- p2ShowMargins
    p1Regrets = p1Objectives .- p1Objectives[di1]
    p2Regrets = p2Objectives .- p2Objectives[di2]

    for drow in h1Rows
        nrow = drow - hi1[1] + 1
        (nrow == di1) && (continue)
        db.dealerregret[drow] = (db.dealerplaycount[drow] * db.dealerregret[drow] + p1Regrets[nrow])
        db.dealerplaycount[drow] += 1
        db.dealerregret[drow] /= db.dealerplaycount[drow]
    end
    for drow in h2Rows
        nrow = drow - hi2[1] + 1
        (nrow == di2) && (continue)
        db.poneregret[drow] = (db.poneplaycount[drow] * db.poneregret[drow] + p2Regrets[nrow])
        db.poneplaycount[drow] += 1
        db.poneregret[drow] /= db.poneplaycount[drow]
    end
    db.dealerprofile[h1Rows] = max.(db.dealerregret[h1Rows], 0.0) ./ sum(max.(db.dealerregret[h1Rows], 0.0))
    db.poneprofile[h2Rows] = max.(db.poneregret[h2Rows], 0.0) ./ sum(max.(db.poneregret[h2Rows], 0.0))

    olddealerprobs = db.dealerplayprob[h1Rows]
    oldponeprobs = db.poneplayprob[h2Rows]

    db.dealerplayprob[h1Rows] = view(db, h1Rows, :dealerprofile) .* view(db, h1Rows, :prob)
    db.poneplayprob[h2Rows] = view(db, h2Rows, :poneprofile) .* view(db, h2Rows, :prob)

    for (ii, ph) in enumerate(p1PlayHands)
        phDealerProbs[counter(ph)] += (db.dealerplayprob[ii] - olddealerprobs[ii])
    end
    for (ii, ph) in enumerate(p2PlayHands)
        phPoneProbs[counter(ph)] += (db.poneplayprob[ii] - oldponeprobs[ii])
    end





end

function threadedCFR(h1Cards::Vector{Card}, h2Cards::Vector{Card}, turncard::Card, 
                    dblock::ReentrantLock, dealerlock::ReentrantLock, ponelock::ReentrantLock)

    (h1, sp1) = canonicalize(h1Cards)
    (h2, sp2) = canonicalize(h2Cards)

    hi1 = handID[h1]
    h1rows = hi1[1]:hi1[2]
    n1 = hi1[2] - hi1[1] + 1
    hi2 = handID[h2]
    h2rows = hi2[1]:hi2[2]
    n2 = hi2[2] - hi2[1] + 1



    df1 = nothing
    df2 = nothing 
    lock(dblock)
    try
        df1 = db[h1rows, :]         # not @view because db may be updated in the background. This is a slowdown of ~150μs per deal
        df2 = db[h2rows, :]
    finally
        unlock(dblock)
    end



    d1cards = [unCanonicalize(d, sp1) for d in @view df1[:, :discard]]
    d2cards = [unCanonicalize(d, sp2) for d in @view df2[:, :discard]]

    d1ranks = [stripsuits(d) for d in @view df1[:, :discard]]
    d2ranks = [stripsuits(d) for d in @view df2[:, :discard]]

    p1playhands = @view df1[:, :playhand]
    p2playhands = @view df2[:, :playhand]

    p1weights = ProbabilityWeights(@view df1[:, :dealerprofile])
    p2weights = ProbabilityWeights(@view df2[:, :poneprofile])

    di1 = sample(1:n1, p1weights)
    di2 = sample(1:n2, p2weights)


    p1playmargins = [threadednaiveplay([[p1h...], [p2playhands[di2]...]], [p1d, d2ranks[di2]], turncard.rank, dealerlock, ponelock)[1] for (p1h, p1d) in zip(p1playhands, d1ranks)]
    p2playmargins = [threadednaiveplay([[p1playhands[di1]...], [p2h...]], [d1ranks[di1], p2d], turncard.rank, dealerlock, ponelock)[1] for (p2h, p2d) in zip(p2playhands, d2ranks)]


    p1showhands = [setdiff(h1Cards, d) for d in d1cards]
    p2showhands = [setdiff(h2Cards, d) for d in d2cards]

    p1cribs = [vcat(d1, d2cards[di2]) for d1 in d1cards]
    p2cribs = [vcat(d1cards[di1], d2) for d2 in d2cards]

    p1showscores = [scoreShow(H, turncard) for H in p1showhands]
    p2showscores = [scoreShow(H, turncard) for H in p2showhands]

    p1cribscores = [scoreShow(C, turncard, isCrib = true) for C in p1cribs]
    p2cribscores = [scoreShow(C, turncard, isCrib = true) for C in p2cribs]

    p1showmargins = [ss - p2showscores[di2] for ss in p1showscores] .+ p1cribscores
    p2showmargins = [p1showscores[di1] - ss for ss in p2showscores] .+ p2cribscores

    p1objectives = zeros(Int64, n1)
    p2objectives = zeros(Int64, n2)
    try
        p1objectives = p1playmargins .+ p1showmargins
        p2objectives = -p2playmargins .- p2showmargins
    catch
        @infiltrate()
    end

    p1regrets = p1objectives .- p1objectives[di1]
    p2regrets = p2objectives .- p2objectives[di2]


    olddealerprobs = nothing
    oldponeprobs = nothing
    newdealerprobs = nothing
    newponeprobs = nothing
    lock(dblock)
    try
        for (nrow, drow) in enumerate(h1rows)
            (nrow == di1) && continue
            db.dealerregret[drow] = (db.dealerplaycount[drow] * db.dealerregret[drow] + p1regrets[nrow])
            db.dealerplaycount[drow] += 1
            db.dealerregret[drow] /= db.dealerplaycount[drow]
        end
        for (nrow, drow) in enumerate(h2rows)
            (nrow == di2) && (continue)
            db.poneregret[drow] = (db.poneplaycount[drow] * db.poneregret[drow] + p2regrets[nrow])
            db.poneplaycount[drow] += 1
            db.poneregret[drow] /= db.poneplaycount[drow]
        end
        db.dealerprofile[h1rows] = max.(db.dealerregret[h1rows], 0.0) ./ sum(max.(db.dealerregret[h1rows], 0.0))
        db.poneprofile[h2rows] = max.(db.poneregret[h2rows], 0.0) ./ sum(max.(db.poneregret[h2rows], 0.0))

        olddealerprobs = db.dealerplayprob[h1rows]
        oldponeprobs = db.poneplayprob[h2rows]

        db.dealerplayprob[h1rows] = view(db, h1rows, :dealerprofile) .* view(db, h1rows, :prob)
        db.poneplayprob[h2rows] = view(db, h2rows, :poneprofile) .* view(db, h2rows, :prob)

        newdealerprobs = db.dealerplayprob[h1rows]
        newponeprobs = db.poneplayprob[h2rows]
    finally
        unlock(dblock)
    end

    lock(dealerlock)
    try
        for (ii, ph) in enumerate(p1playhands)
            phDealerProbs[counter(ph)] += (newdealerprobs[ii] - olddealerprobs[ii])
        end
    finally
        unlock(dealerlock)
    end

    lock(ponelock)
    try
        for (ii, ph) in enumerate(p2playhands)
            phPoneProbs[counter(ph)] += (newponeprobs[ii] - oldponeprobs[ii])
        end
    finally
        unlock(ponelock)
    end



end

