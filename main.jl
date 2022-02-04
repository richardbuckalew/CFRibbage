using Combinatorics, Serialization, IterTools, ProgressMeter, StatsBase, ProfileView, BenchmarkTools
using DataStructures, Random, ThreadTools

include("dbUtils.jl");
include("playUtils.jl");

(@isdefined cardsuits) || (const cardsuits = collect(1:4));
(@isdefined cardranks) || (const cardranks = collect(1:13));
(@isdefined cardvalues) || (const cardvalues = vcat(collect(1:10), [10, 10, 10]));
(@isdefined suitnames) || (const suitnames = ["spades", "hearts", "diamonds", "clubs"]);
(@isdefined ranknames) || (const ranknames = ["ace", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "jack", "queen", "king"]);
(@isdefined shortsuit) || (const shortsuit = ["♠","♡","♢","♣"]);
(@isdefined shortrank) || (const shortrank = ["A","2","3","4","5","6","7","8","9","T","J","Q","K"]);

struct Card
    rank::Int64
    suit::Int64    
end
Base.show(io::IO, c::Card) = print(io, shortname(c));
showHand(H) = display(permutedims(H));
Base.isless(x::Card, y::Card) = (x.suit > y.suit) ? true : (x.rank < y.rank);   ## Bridge bidding suit order, ace == 1

(@isdefined standardDeck) || (const standardDeck = [Card(r,s) for r in cardranks for s in cardsuits]);
(@isdefined rankDeck) || (const rankDeck = [c.rank for c in standardDeck]);

fullname(c::Card) = ranknames[c.rank] * " of " * suitnames[c.suit];
shortname(c::Card) = shortrank[c.rank] * shortsuit[c.suit];
handname(h::AbstractArray{Card}) = permutedims(shortname.(sort(h, by = c->c.rank)));

handranks(h::Vector{Card}) = sort([c.rank for c in h]);
handsuits(h::Vector{Card}) = sort([c.suit for c in h]);


function canonicalize(h::Vector{Card})
    H = [[c.rank for c in h if c.suit == ii] for ii in 1:4];
    sort!.(H);  # sort each suit
    sp1 = sortperm(H);   # sort suits lexicographically
    H = H[sp1];
    sp2 = sortperm(H, by = length, rev = true);  # subsort by length, longest to shortest
    H = H[sp2];
    sp = sp1[sp2]; # suit permutation (for reconstructing h)
    return (Tuple(Tuple.(H)), sp);
end

function unCanonicalize(h, sp)
    hand = Card[];
    for (ii, s) in enumerate(sp)
        for r in h[ii]
            (r == 0) && continue;
            push!(hand, Card(r, s));
        end
    end
    return hand
end

function scoreShow(h::Vector{Card}, turn::Card; isCrib = false)

    cCombs = collect.([combinations(vcat(h,[turn]), ii) for ii in 2:(length(h)+1)]);

    ranks = sort(vcat(handranks(h), handranks([turn])));
    rCombs = [[handranks(c) for c in comb] for comb in cCombs];

    nPairs = count([x[1] == x[2] for x in rCombs[1]]);
    pairScore = 2 * nPairs;
    # display("pairs: " * string(pairScore))

    nRuns = 0;
    runLength = 0;
    for ell in (length(ranks)-1):-1:2
        runLength = ell + 1;
        adjRanks = [r .- minimum(r) .+ 1 for r in rCombs[ell]]
        nRuns = count(map(isequal(collect(1:runLength)), adjRanks));
        (nRuns > 0) && break;
    end
    runScore = nRuns * runLength;
    # display("runs: " * string(runScore))
    
    vCombs = [[cardvalues[c] for c in comb] for comb in rCombs];
    S = sum.(vcat(vCombs...));
    nFifteens = count(S .== 15);
    fifteenScore = 2 * nFifteens;
    # display("fifteens: " * string(fifteenScore));


    if isCrib
        suits = sort([c.suit for c in h]);
    else
        suits = sort([c.suit for c in vcat(h, [turn])]);
    end
    flushScore = 0;
    for ell in length(suits):-1:4
        sCombs = collect(combinations(suits, ell));
        sLens = length.(Set.(sCombs))
        if any(sLens .== 1)
            flushScore = ell;
            break;
        end
    end
    # display("flush: " * string(flushScore));

    return pairScore + runScore + fifteenScore + flushScore


end

function dealHands(deck = standardDeck, handSize = 6)
    dealtCards = sample(deck, handSize * 2 + 1, replace = false);
    h1 = dealtCards[1:handSize];
    h2 = dealtCards[handSize+1:2*handSize];
    turn = dealtCards[end];
    return (h1, h2, turn);
end

function resolveplay(h1, h2, phID, M)
    return M[phID[h1], phID[h2]];
end

function makeCrib(d1, d2)
    return vcat(d1, d2);
end

function doCFR(db, hID, phID, M)

    (h1Cards, h2Cards, turnCard) = dealHands();

    (h1, sp1) = canonicalize(h1Cards);
    (h2, sp2) = canonicalize(h2Cards);

    hi1 = hID[h1];
    h1Rows = hi1[1]:hi1[2];
    hi2 = hID[h2];
    h2Rows = hi2[1]:hi2[2];

    allD1 = db[h1Rows, :discard];
    allD1Cards = [unCanonicalize(d, sp1) for d in allD1];
    allD2 = db[h2Rows, :discard];
    allD2Cards = [unCanonicalize(d, sp2) for d in allD2];
    
    p1PlayHands = db[h1Rows, :playhand];
    p2PlayHands = db[h2Rows, :playhand];

    p1weights = ProbabilityWeights(db[h1Rows, :dealerprofile]);
    p2weights = ProbabilityWeights(db[h2Rows, :poneprofile]);
    di1 = sample(1:length(allD1), p1weights);
    di2 = sample(1:length(allD2), p2weights);

    p1playhand = p1PlayHands[di1];
    p2playhand = p2PlayHands[di2];

    p1PlayMargins = [resolveplay(p1h, p2playhand, phID, M) for p1h in p1PlayHands];
    p2PlayMargins = [resolveplay(p1playhand, p2h, phID, M) for p2h in p2PlayHands];

    
    p1ShowHands = [setdiff(h1Cards, d) for d in allD1Cards];
    p2ShowHands = [setdiff(h2Cards, d) for d in allD2Cards];

    p1Cribs = [makeCrib(d1, allD2Cards[di2]) for d1 in allD1Cards];
    p2Cribs = [makeCrib(allD1Cards[di1], d2) for d2 in allD2Cards];

    p1ShowScores = [scoreShow(H, turnCard) for H in p1ShowHands];
    p1CribScores = [scoreShow(C, turnCard, isCrib = true) for C in p1Cribs];
    p2ShowScores = [scoreShow(H, turnCard) for H in p2ShowHands];
    p2CribScores = [scoreShow(C, turnCard, isCrib = true) for C in p2Cribs];
    p1ShowMargins = [ss - p2ShowScores[di2] for ss in p1ShowScores] .+ p1CribScores;     # actual scores: maximize this
    p2ShowMargins = [p1ShowScores[di1] - ss for ss in p2ShowScores] .+ p2CribScores;     # actual scores: minimize this

    p1Objectives = p1PlayMargins .+ p1ShowMargins;
    p2Objectives = -p2PlayMargins .- p2ShowMargins;
    p1Regrets = p1Objectives .- p1Objectives[di1];
    p2Regrets = p2Objectives .- p2Objectives[di2];

    for drow in h1Rows
        nrow = drow - hi1[1] + 1;
        (nrow == di1) && (continue);
        db.dealerregret[drow] = (db.dealerplaycount[drow] * db.dealerregret[drow] + p1Regrets[nrow]);
        db.dealerplaycount[drow] += 1;
        db.dealerregret[drow] /= db.dealerplaycount[drow];
    end
    for drow in h2Rows
        nrow = drow - hi2[1] + 1;
        (nrow == di2) && (continue);
        db.poneregret[drow] = (db.poneplaycount[drow] * db.poneregret[drow] + p2Regrets[nrow]);
        db.poneplaycount[drow] += 1;
        db.poneregret[drow] /= db.poneplaycount[drow];
    end
    db.dealerprofile[h1Rows] = max.(db.dealerregret[h1Rows], 0.0) ./ sum(max.(db.dealerregret[h1Rows], 0.0));
    db.poneprofile[h2Rows] = max.(db.poneregret[h2Rows], 0.0) ./ sum(max.(db.poneregret[h2Rows], 0.0));


    # display(db[h1Rows, :])
    # display(db[h2Rows, :])


end



# (@isdefined db) || (const db = deserialize("db.jls"));
# (@isdefined handID) || (const handID = deserialize("handID.jls"));
# (@isdefined playHandRowID) || (const playHandRowID = deserialize("playHandRowID.jls"));
# (@isdefined playHandID) || (const playHandID = deserialize("playHandID.jls"));
# (@isdefined perfectSolveMatrix) || (const perfectSolveMatrix = deserialize("perfectSolveMatrix.jls"));


# doCFR(db, handID, playHandID, perfectSolveMatrix)