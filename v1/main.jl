using BenchmarkTools, Combinatorics, Serialization, IterTools, StatsBase, DataStructures, Random, DataFrames, Folds, FLoops, ThreadTools, LinearAlgebra


struct Card
    rank::Int64;
    suit::Int64;    
end
Base.show(io::IO, c::Card) = print(io, shortname(c))
showHand(H) = display(permutedims(H))
Base.isless(x::Card, y::Card) = (x.suit > y.suit) ? true : (x.rank < y.rank)   ## Bridge bidding suit order, ace == 1


function countertovector(c::Accumulator{Int64, Int64})
    return sort([r for r in keys(c) for ii in 1:c[r]])
end



(@isdefined cardsuits) || (const cardsuits = collect(1:4))
(@isdefined cardranks) || (const cardranks = collect(1:13))
(@isdefined testsuits) || (const testsuits = [1, 2])
(@isdefined testranks) || (const testranks = collect(1:7))
(@isdefined cardvalues) || (const cardvalues = vcat(collect(1:10), [10, 10, 10]))
(@isdefined suitnames) || (const suitnames = ["spades", "hearts", "diamonds", "clubs"])
(@isdefined ranknames) || (const ranknames = ["ace", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "jack", "queen", "king"])
(@isdefined shortsuit) || (const shortsuit = ["♠","♡","♢","♣"])
(@isdefined shortrank) || (const shortrank = ["A","2","3","4","5","6","7","8","9","T","J","Q","K"])

(@isdefined PP) || (const PP = [
    [[1]],
    [[2], [1,1]],
    [[3], [2,1], [1,2], [1,1,1]],
    [[4], [3,1], [1,3], [2,2], [2,1,1], [1,2,1], [1,1,2], [1,1,1,1]]
]);
(@isdefined cardcombs) || (const cardcombs = [collect(combinations(cardranks, n)) for n in 1:4])


(@isdefined standardDeck) || (const standardDeck = [Card(r,s) for r in cardranks for s in cardsuits])
(@isdefined testDeck) || (const testDeck = [Card(r,s) for r in testranks for s in testsuits])
(@isdefined rankDeck) || (const rankDeck = [c.rank for c in standardDeck])

(@isdefined fullname) || (const fullname(c::Card) = ranknames[c.rank] * " of " * suitnames[c.suit])
(@isdefined shortname) || (const shortname(c::Card) = shortrank[c.rank] * shortsuit[c.suit])
(@isdefined handname) || (const handname(h::AbstractArray{Card}) = permutedims(shortname.(sort(h, by = c->c.rank))))

(@isdefined handranks) || (const handranks(h::Vector{Card}) = sort([c.rank for c in h]))
(@isdefined handsuits) || (const handsuits(h::Vector{Card}) = sort([c.suit for c in h]))


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


struct PlayRecord
    dealerhand::handType
    poneHand::handType
    dealerdiscardindex::Int64
    ponediscardindex::Int64
    dealerdelta::Float64
    ponedelta::Float64
end

function dealHands(deck = standardDeck, handSize = 6)
    dealtCards = sample(deck, handSize * 2 + 1, replace = false)
    h1 = dealtCards[1:handSize]
    h2 = dealtCards[handSize+1:2*handSize]
    turn = dealtCards[end]
    return (h1, h2, turn)
end

include("playUtils.jl")
include("dbUtils.jl")
include("analysisUtils.jl")


(@isdefined allhands) || (@time const allhands = deserialize("allhands.jls"))
(@isdefined hRows) || (@time const hRows = deserialize("hRows.jls"))  # Tuple => NTuple{2, Int64}
(@isdefined allPH) || (@time const allPH = deserialize("allPH.jls"))     # Vector{Counter}
(@isdefined phID) || (@time const phID = deserialize("phID.jls"))        # Counter => Int64
(@isdefined phRows) || (@time const phRows = deserialize("phRows.jls"))  # Counter => Vector{Int64}
(@isdefined M) || (@time const M = Folds.map(unpackflat, deserialize("M.jls")))

(@isdefined nph) || (const nph = length(allPH))
(@isdefined BVinclude) || (const BVinclude = makebvs(issubhand, cardranks, cardsuits))
(@isdefined BVexclude) || (const BVexclude = makebvs(excludes, cardranks, cardsuits))





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


function makeCrib(d1, d2)
    return vcat(d1, d2)
end

function stripsuits(D::discardType)::Vector{Int64}
    return vcat([[x...] for x in D]...);
end



function CFR(h1cards, h2cards, turncard, db, phDealerProbs, phPoneProbs, reusablemodels)

    (h1, sp1) = canonicalize(h1cards)
    (h2, sp2) = canonicalize(h2cards)


    h1rows = range(hRows[h1]...)
    h2rows = range(hRows[h2]...)
    n1 = length(h1rows)
    n2 = length(h2rows)

    df1 = db[h1rows, :]
    df2 = db[h2rows, :]      

    p1played = maximum(df1.dealerplaycount)
    p2played = maximum(df2.poneplaycount)

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

    olddealerprobs = df1.dealerplayprob
    oldponeprobs = df2.poneplayprob


    p1showhand = setdiff(h1cards, d1cards[di1])
    p1showscore = scoreShow(p1showhand, turncard)
    p2showhand = setdiff(h2cards, d2cards[di2])
    p2showscore = scoreShow(p2showhand, turncard)

    if p1played < 40

        p1playresults = Folds.collect(play([[p1h...], [p2playhands[di2]...]], [p1d, d2ranks[di2]], turncard.rank, phDealerProbs, phPoneProbs, reusablemodels[1][ii]) for (ii, (p1h, p1d)) in enumerate(zip(p1playhands, d1ranks)))
        p1playmargins = [result[1] for result in p1playresults]
        p1showhands = [setdiff(h1cards, d) for d in d1cards]
        p1cribs = [vcat(d1, d2cards[di2]) for d1 in d1cards]
        p1showscores = [scoreShow(H, turncard) for H in p1showhands]
        p1cribscores = [scoreShow(C, turncard, isCrib = true) for C in p1cribs]
        p1showmargins = [ss - p2showscore for ss in p1showscores] .+ p1cribscores
        p1objectives = p1playmargins .+ p1showmargins
        p1regrets = p1objectives .- p1objectives[di1]

        for nrow in 1:n1
            (nrow == di1) && continue
            df1.dealerregret[nrow] = df1.dealerplaycount[nrow] * df1.dealerregret[nrow] + p1regrets[nrow]
            df1.dealerplaycount[nrow] += 1
            df1.dealerregret[nrow] /= df1.dealerplaycount[nrow]
        end
        if any(df1.dealerregret .> 0)
            df1.dealerprofile = max.(df1.dealerregret, 0.0) ./ sum(max.(df1.dealerregret, 0.0))
        else
            df1.dealerprofile .= 1 / n1
        end

    else

        activerows = df1.dealerprofile .> 0
        nactive = sum(activerows)
        if nactive > 1

            margins = zeros(Int64, n1)

            @floop for nrow in 1:n1
                activerows[nrow] || continue

                playresult = play([[p1playhands[nrow]...], [p2playhands[di2]...]], [d1ranks[nrow], d2ranks[di2]], turncard.rank, phDealerProbs, phPoneProbs, reusablemodels[1][nrow])
                playmargin = playresult[1]
                showhand = setdiff(h1cards, d1cards[nrow])
                showscore = scoreShow(showhand, turncard)
                crib = vcat(d1cards[nrow], d2cards[di2])
                cribscore = scoreShow(crib, turncard, isCrib = true)
                showmargin = showscore - p2showscore + cribscore
                margins[nrow] = playmargin + showmargin
                
            end

            regrets = margins .- margins[di1]

            for nrow in 1:n1
                activerows[nrow] || continue
                df1.dealerregret[nrow] = df1.dealerplaycount[nrow] * df1.dealerregret[nrow] + regrets[nrow]
                df1.dealerplaycount[nrow] += 1
                df1.dealerregret[nrow] /= df1.dealerplaycount[nrow]
            end
            if any(df1.dealerregret .> 0)
                df1.dealerprofile = max.(df1.dealerregret, 0.0) ./ sum(max.(df1.dealerregret, 0.0))
            else
                df1.dealerprofile .= 1 / n1
            end

        end

    end



    if p2played < 40    

        p2playresults = Folds.collect(play([[p1playhands[di1]...], [p2h...]], [d1ranks[di1], p2d], turncard.rank, phDealerProbs, phPoneProbs, reusablemodels[2][ii]) for (ii, (p2h, p2d)) in enumerate(zip(p2playhands, d2ranks)))
        p2playmargins = [result[1] for result in p2playresults]
        p2showhands = [setdiff(h2cards, d) for d in d2cards]
        p2cribs = [vcat(d1cards[di1], d2) for d2 in d2cards]
        p2showscores = [scoreShow(H, turncard) for H in p2showhands]
        p2cribscores = [scoreShow(C, turncard, isCrib = true) for C in p2cribs]
        p2showmargins = [p1showscore - ss for ss in p2showscores] .+ p2cribscores
        p2objectives = -p2playmargins .- p2showmargins
        p2regrets = p2objectives .- p2objectives[di2]

        for nrow in 1:n2
            (nrow == di2) && continue
            df2.poneregret[nrow] = df2.poneplaycount[nrow] * df2.poneregret[nrow] + p2regrets[nrow]
            df2.poneplaycount[nrow] += 1
            df2.poneregret[nrow] /= df2.poneplaycount[nrow]
        end
        if any(df2.poneregret .> 0)
            df2.poneprofile = max.(df2.poneregret, 0.0) ./ sum(max.(df2.poneregret, 0.0))
        else
            df2.poneprofile .= 1 / n2
        end

    else

        activerows = df2.poneprofile .> 0
        nactive = sum(activerows)
        if nactive > 1

            margins = zeros(Int64, n2)

            @floop for nrow in 1:n2
                activerows[nrow] || continue

                playresult = play([[p1playhands[di1]...], [p2playhands[nrow]...]], [d1ranks[di1], d2ranks[nrow]], turncard.rank, phDealerProbs, phPoneProbs, reusablemodels[2][nrow])
                playmargin = playresult[1]
                showhand = setdiff(h2cards, d2cards[nrow])
                showscore = scoreShow(showhand, turncard)
                crib = vcat(d1cards[di1], d2cards[nrow])
                cribscore = scoreShow(crib, turncard, isCrib = true)
                showmargin = p1showscore - showscore + cribscore
                margins[nrow] = playmargin + showmargin

            end

            regrets = -margins .+ margins[di2]

            for nrow in 1:n2
                activerows[nrow] || continue
                df2.poneregret[nrow] = df2.poneplaycount[nrow] * df2.poneregret[nrow] + regrets[nrow]
                df2.poneplaycount[nrow] += 1
                df2.poneregret[nrow] /= df2.poneplaycount[nrow]
            end
            if any(df2.poneregret .> 0)
                df2.poneprofile = max.(df2.poneregret, 0.0) ./ sum(max.(df2.poneregret, 0.0))
            else
                df2.poneprofile .= 1 / n2
            end

        end

    end



    newdealerprobs = df1.dealerprofile .* df1.prob
    newponeprobs = df2.poneprofile .* df2.prob

    db[h1rows, :dealerplaycount] = df1.dealerplaycount
    db[h1rows, :dealerregret] = df1.dealerregret
    db[h1rows, :dealerprofile] = df1.dealerprofile   
    db[h1rows, :dealerplayprob] = newdealerprobs
    db[h2rows, :poneplaycount] = df2.poneplaycount
    db[h2rows, :poneregret] = df2.poneregret
    db[h2rows, :poneprofile] = df2.poneprofile     
    db[h2rows, :poneplayprob] = newponeprobs

    for (ii, ph) in enumerate(p1playhands)
        phDealerProbs[counter(ph)] += newdealerprobs[ii] - olddealerprobs[ii]
    end        
    for (ii, ph) in enumerate(p2playhands)
        phPoneProbs[counter(ph)] += newponeprobs[ii] - oldponeprobs[ii]
    end    
        

    return PlayRecord(h1, h2, di1, di2, norm(newdealerprobs-olddealerprobs, Inf), norm(newponeprobs-oldponeprobs, Inf))


end


function CFRbackup(h1cards, h2cards, turncard, db, phDealerProbs, phPoneProbs, reusablemodels)

    (h1, sp1) = canonicalize(h1cards)
    (h2, sp2) = canonicalize(h2cards)


    h1rows = range(hRows[h1]...)
    h2rows = range(hRows[h2]...)
    n1 = length(h1rows)
    n2 = length(h2rows)

    df1 = db[h1rows, :]
    df2 = db[h2rows, :]      

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

    # p1playresults = [play([[p1h...], [p2playhands[di2]...]], [p1d, d2ranks[di2]], turncard.rank, phDealerProbs, phPoneProbs) for (p1h, p1d) in zip(p1playhands, d1ranks)]
    # p2playresults = [play([[p1playhands[di1]...], [p2h...]], [d1ranks[di1], p2d], turncard.rank, phDealerProbs, phPoneProbs) for (p2h, p2d) in zip(p2playhands, d2ranks)]

    p1playresults = Folds.collect(play([[p1h...], [p2playhands[di2]...]], [p1d, d2ranks[di2]], turncard.rank, phDealerProbs, phPoneProbs, reusablemodels[1][ii]) for (ii, (p1h, p1d)) in enumerate(zip(p1playhands, d1ranks)))
    p2playresults = Folds.collect(play([[p1playhands[di1]...], [p2h...]], [d1ranks[di1], p2d], turncard.rank, phDealerProbs, phPoneProbs, reusablemodels[2][ii]) for (ii, (p2h, p2d)) in enumerate(zip(p2playhands, d2ranks)))


    p1playmargins = [result[1] for result in p1playresults]
    p2playmargins = [result[1] for result in p2playresults]

    p1showhands = [setdiff(h1cards, d) for d in d1cards]
    p2showhands = [setdiff(h2cards, d) for d in d2cards]

    p1cribs = [vcat(d1, d2cards[di2]) for d1 in d1cards]
    p2cribs = [vcat(d1cards[di1], d2) for d2 in d2cards]

    p1showscores = [scoreShow(H, turncard) for H in p1showhands]
    p2showscores = [scoreShow(H, turncard) for H in p2showhands]

    p1cribscores = [scoreShow(C, turncard, isCrib = true) for C in p1cribs]
    p2cribscores = [scoreShow(C, turncard, isCrib = true) for C in p2cribs]

    p1showmargins = [ss - p2showscores[di2] for ss in p1showscores] .+ p1cribscores
    p2showmargins = [p1showscores[di1] - ss for ss in p2showscores] .+ p2cribscores

    p1objectives = p1playmargins .+ p1showmargins
    p2objectives = -p2playmargins .- p2showmargins

    p1regrets = p1objectives .- p1objectives[di1]
    p2regrets = p2objectives .- p2objectives[di2]

    olddealerprobs = df1.dealerplayprob
    oldponeprobs = df2.poneplayprob

    for nrow in 1:n1
        (nrow == di1) && continue
        df1.dealerregret[nrow] = df1.dealerplaycount[nrow] * df1.dealerregret[nrow] + p1regrets[nrow]
        df1.dealerplaycount[nrow] += 1
        df1.dealerregret[nrow] /= df1.dealerplaycount[nrow]
    end
    if any(df1.dealerregret .> 0)
        df1.dealerprofile = max.(df1.dealerregret, 0.0) ./ sum(max.(df1.dealerregret, 0.0))
    else
        df1.dealerprofile .= 1 / n1
    end
    newdealerprobs = df1.dealerprofile .* df1.prob

    for nrow in 1:n2
        (nrow == di2) && continue
        df2.poneregret[nrow] = df2.poneplaycount[nrow] * df2.poneregret[nrow] + p2regrets[nrow]
        df2.poneplaycount[nrow] += 1
        df2.poneregret[nrow] /= df2.poneplaycount[nrow]
    end
    if any(df2.poneregret .> 0)
        df2.poneprofile = max.(df2.poneregret, 0.0) ./ sum(max.(df2.poneregret, 0.0))
    else
        df2.poneprofile .= 1 / n2
    end
    newponeprobs = df2.poneprofile .* df2.prob

    db[h1rows, :dealerplaycount] = df1.dealerplaycount
    db[h1rows, :dealerregret] = df1.dealerregret
    db[h1rows, :dealerprofile] = df1.dealerprofile   
    db[h1rows, :dealerplayprob] = newdealerprobs
    db[h2rows, :poneplaycount] = df2.poneplaycount
    db[h2rows, :poneregret] = df2.poneregret
    db[h2rows, :poneprofile] = df2.poneprofile     
    db[h2rows, :poneplayprob] = newponeprobs

    for (ii, ph) in enumerate(p1playhands)
        phDealerProbs[counter(ph)] += newdealerprobs[ii] - olddealerprobs[ii]
    end        
    for (ii, ph) in enumerate(p2playhands)
        phPoneProbs[counter(ph)] += newponeprobs[ii] - oldponeprobs[ii]
    end    
        

    return PlayRecord(h1, h2, di1, di2, norm(newdealerprobs-olddealerprobs, Inf), norm(newponeprobs-oldponeprobs, Inf))

    # display(db[h1rows,:])
    # display([phDealerProbs[counter(ph)] for ph in p1playhands])
    # display(db[h2rows,:])
    # display([phPoneProbs[counter(ph)] for ph in p2playhands])

end


function getregrets(h1cards, h2cards, turncard, db, phDealerProbs, phPoneProbs)

    (h1, sp1) = canonicalize(h1cards)
    (h2, sp2) = canonicalize(h2cards)


    h1rows = range(hRows[h1]...)
    h2rows = range(hRows[h2]...)
    n1 = length(h1rows)
    n2 = length(h2rows)

    df1 = db[h1rows, :]
    df2 = db[h2rows, :]  

    p1weights = ProbabilityWeights(@view df1[:, :dealerprofile])
    p2weights = ProbabilityWeights(@view df2[:, :poneprofile])

    di1 = sample(1:n1, p1weights)
    di2 = sample(1:n2, p2weights)

    p1playhand = [df1.playhand[di1]...]
    p2playhand = [df2.playhand[di2]...]

    p1dcards = unCanonicalize(df1.discard[di1], sp1)
    p2dcards = unCanonicalize(df2.discard[di2], sp2)

    p1dranks = [c.rank for c in p1dcards]
    p2dranks = [c.rank for c in p2dcards]

    p1showhand = setdiff(h1cards, p1dcards)
    p2showhand = setdiff(h2cards, p2dcards)

    p1objectives = zeros(Int64, n1)
    p2objectives = zeros(Int64, n2)

    @floop for ii in 1:n1

        dcards = unCanonicalize(df1.discard[ii], sp1)
        dranks = [c.rank for c in dcards]
        playhand = [df1.playhand[ii]...]
        showhand = setdiff(h1cards, dcards)
        crib = vcat(dcards, p2dcards)
        
        p1objectives[ii] = play([playhand, copy(p2playhand)], [dranks, p2dranks], turncard.rank, phDealerProbs, phPoneProbs, [trues(nph), trues(nph)])[1] +
                                (scoreShow(showhand, turncard) + scoreShow(crib, turncard, isCrib = true) - scoreShow(p2showhand, turncard))

    end

    @floop for ii in 1:n2

        dcards = unCanonicalize(df2.discard[ii], sp2)
        dranks = [c.rank for c in dcards]
        playhand = [df2.playhand[ii]...]
        showhand = setdiff(h2cards, dcards)
        crib = vcat(dcards, p1dcards)
        
        p2objectives[ii] = -play([copy(p1playhand), playhand], [p1dranks, dranks], turncard.rank, phDealerProbs, phPoneProbs, [trues(nph), trues(nph)])[1] +
                                (scoreShow(showhand, turncard) - scoreShow(crib, turncard, isCrib = true) - scoreShow(p1showhand, turncard))

    end

    p1regrets = p1objectives .- p1objectives[di1]
    p2regrets = p2objectives .- p2objectives[di2]



    return (di1, di2, p1regrets, p2regrets)

end








@inline function resetmodels!(reusablemodels)
    for ii in (1,2)
        for jj in 1:16
            for kk in 1:2
                for mm in 1:nph
                    reusablemodels[ii][jj][kk][mm] = true
                end
            end
        end
    end
end


function train(nbatches = 20, ndeals = 100000)
    

    display("Loading... ")

    db = deserialize("db.jls")
    phDealerProbs = deserialize("phDealerProbs.jls")
    phPoneProbs = deserialize("phPoneProbs.jls")

    reusablemodels = [[[trues(nph), trues(nph)] for ii in 1:16] for jj in (1,2)]


    records = Vector{PlayRecord}(undef, ndeals*nbatches)

    for nbatch in 1:nbatches
        
        print(nbatch, " ")
        @time begin
            for ndeal in 1:ndeals

                resetmodels!(reusablemodels)
                (h1cards, h2cards, turncard) = dealHands(standardDeck)
                records[ndeal + ndeals * (nbatch-1)] = CFR(h1cards, h2cards, turncard, db, phDealerProbs, phPoneProbs, reusablemodels)

            end

            mv("db.jls", "backup/db.bak", force=true)
            mv("phDealerProbs.jls", "backup/phDealerProbs.bak", force=true)
            mv("phPoneProbs.jls", "backup/phPoneProbs.bak", force=true)

            serialize("db.jls", db)
            serialize("phDealerProbs.jls", phDealerProbs)
            serialize("phPoneProbs.jls", phPoneProbs)

        end
    end
    savesnapshot(db)
    saverecords(records)


end




function testCFR()

    db = deserialize("db.jls")
    phDealerProbs = deserialize("phDealerProbs.jls")
    phPoneProbs = deserialize("phPoneProbs.jls")

    models = [[[trues(nph), trues(nph)] for ii in 1:16] for jj in (1,2)]

    @btime begin
        (h1cards, h2cards, turncard) = dealHands(standardDeck)
        CFR(h1cards, h2cards, turncard, $db, $phDealerProbs, $phPoneProbs, $models)
    end

    # (h1cards, h2cards, turncard) = dealHands(standardDeck)
    # @time CFR(h1cards, h2cards, turncard, db, phDealerProbs, phPoneProbs)

end


function testregret()

    db = deserialize("db.jls")
    phDealerProbs = deserialize("phDealerProbs.jls")
    phPoneProbs = deserialize("phPoneProbs.jls")

    @btime begin
        (h1cards, h2cards, turncard) = dealHands(standardDeck)
        getregrets(h1cards, h2cards, turncard, $db, $phDealerProbs, $phPoneProbs)
    end

end

for ii in 1:100
    train()
end