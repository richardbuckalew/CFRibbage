using Combinatorics, DataFrames, DataStructures, FLoops, Folds, IterTools, ProgressMeter, Serialization, ThreadTools


struct Card
    rank::Int64;
    suit::Int64;    
end
Base.show(io::IO, c::Card) = print(io, shortname(c))
Base.isless(x::Card, y::Card) = (x.suit > y.suit) ? true : (x.rank < y.rank)   ## Bridge bidding suit order, ace == 1


# Hands take one of several forms:
#  - A vector of cards
#  - A canonicalized form, a tuple of tuples, each tuple containing the ranks of a given suit. The canonical form has an
#       associated suit permutation sp for reconstructing the original hand. With the suit perm, it is thus equivalent
#       to the vector of cards. Without the suit perm, it is unique up to the symmetries of cribbage. Thus this form, 
#       *without* the suit perm, is used in the strategy database
#  - An Accumulator, counting the number of repetitions of each rank present in the hand. This form is for the play phase,
#       in which a card's suit is irrelevant. The Accumulator form is convenient for the recursive play algorithms.
#
# By convention, the first form will be denoted by variables containing the full string 'hand'. The second will be denoted
#   by a lowercase h, and the third by an upper case H.
#
#
# NOTE: we have hand-coded the assumption that there are four suits.

const handType = Vector{Card}
const hType = Union{
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
const HType = Accumulator{Int64, Int64}


"Convert a vector of Cards into its canonical representation."
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


"Convert an accumulator hand into a vector of ranks"
function c2v(H::HType)
    return sort([r for r in keys(H) for ii in 1:H[r]])
end


(@isdefined cardsuits) || (const cardsuits = collect(1:4))
(@isdefined cardranks) || (const cardranks = collect(1:13))
(@isdefined cardvalues) || (const cardvalues = vcat(collect(1:10), [10, 10, 10]))

(@isdefined suitnames) || (const suitnames = ["spades", "hearts", "diamonds", "clubs"])
(@isdefined ranknames) || (const ranknames = ["ace", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "jack", "queen", "king"])
(@isdefined shortsuit) || (const shortsuit = ["♠","♡","♢","♣"])
(@isdefined shortrank) || (const shortrank = ["A","2","3","4","5","6","7","8","9","T","J","Q","K"])
(@isdefined fullname) || (const fullname(c::Card) = ranknames[c.rank] * " of " * suitnames[c.suit])
(@isdefined shortname) || (const shortname(c::Card) = shortrank[c.rank] * shortsuit[c.suit])

(@isdefined standardDeck) || (const standardDeck = [Card(r,s) for r in cardranks for s in cardsuits])


"Deal every hand from a deck and count repetitions of canonical forms"
function dealAllHands(deck::Vector{Card})
    hCounts = counter(hType)
    @showprogress 1 for comb in combinations(deck, 6)
        h, = canonicalize(comb)
        inc!(hCounts, h)
    end
    return hCounts
end


const dType = Union{
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


"Get all distinct discards for hand h in canonical form. Preserves suit permutation."
function getDiscards(h::hType)
    D = []

    # choose two distinct suits to discard from
    for (i1, i2) in combinations(1:4, 2)
        s1 = h[i1]
        s2 = h[i2]
        (isempty(s1) || isempty(s2)) && continue

        # If the underlying hand has symmetry, then some discards have symmetry as well.
        if s1 != s2
            i1 = findfirst(Ref(s1) .== h)
            i2 = findfirst(Ref(s2) .== h)
        end

        #choose a card from each suit
        for (c1, c2) in product(s1, s2)
            d = [[], [], [], []]
            d[i1] = [c1,]
            d[i2] = [c2,]
            push!(D, Tuple(Tuple.(d)))      # TODO: there must be a faster way to construct this
        end
    end

    # choose one suit to discard 2 cards from. Only do once if multiple suits are equivalent
    for s in unique(h)
        isempty(s) && continue
        (length(s) < 2) && continue

        # for repeatability, always discard from the *first* of several equivalent suits
        i = findfirst(Ref(s) .== h)
        for (c1, c2) in combinations(s, 2)
            d = [[], [], [], []]
            d[i] = [c1, c2]
            push!(D, Tuple(Tuple.(d)))      # TODO: there must be a faster way to construct this
        end
    end

    return Vector{dType}(unique(D))
end


"Get the sorted ranks of the hand resulting from discarding d from h."
function getPlayHand(h, d)
    return Tuple(sort(vcat([Array(setdiff(h[i], d[i])) for i in 1:4]...)))
end




# The core object is a dataframe df containing the CFR data. Each row corresponds to one possible discard from one possible h.
# The hands h are *not* stored in df. Instead, we keep a separate dict hRows which maps each h to a range of rows. This is for
#    efficiency, since this mapping will never change.
#  - p_deal: the probability of being dealt h
#  - discard: the canonical form of each viable discard.
#  - playhand: a tuple of the ranks resulting from making the corresponding discard. Redundant info.
#  - count_dealer, count_pone: a count of how many times this discard has been chosen by the player in past games
#  - dealt_dealer, dealt_pone: a count of how many times this hand has been dealt. Only first row in each hRows block updated.
#  - regret_dealer, regret_pone: the cumulative regret from having not played this discard
#  - profile_dealer, profile_pone: the calculated strategy profile for this hand, based on regret and play count. For each h,
#      this column will sum to 1. Therefore the total column sums to the number of distinct hands h.
#  - p_play_dealer, p_play_pone: The overall strategy profile for this h + d combo. This column sums to 1.
#
# In addition to the dataframe and its row mapping hRows, there are several other supporting data structures:
#  - HRows: a dict which maps *accumulator* hands H to a list of the rows which lead to that H. Since the same H can result
#      from many different initial hands, the list is long and not contiguous (thus it is not a UnitRange).
#  - allh: a list of all distinct hands h. Equivalent to keys(hRows)
#  - allH: a list of all accumulator hands H. Equivalent (but not necessarily in sorted order) to keys(HRows) and keys(HID)
#  - hID: a dict mapping canonical hands h to sequential integer indices. Used for analytics.
#  - HID: A dict mapping accumulator hands H to sequential integer indices. Used for efficiently indexing M.
#  - Hprobs_dealer, Hprobs_pone: this is the probability of a player having a certain accumulator play hand H, based on the
#      combined probability of being dealt a hand and then choosing a discard that can lead to it. It is a sum of values
#      in the profile_xxx column at the indices given by HID[H]. This is stored separately for efficiency, because 1) this
#      value is needed often in the play solver and 2) once they are initially calculated, the values can be updated more easily
#      than by recalculating the sums. This is because after a single training hand, only the probabilities for play hands 
#      reachable from the dealt hand will change. For more details of this update step, see the core training loop.

"Build a new strategy dataframe, and its supporting structures, from a deck."
function buildDB(deck)

    hCounts = dealAllHands(deck)
    N = sum(values(hCounts))

    p_deal = Float64[]
    discard = dType[]
    playhand = NTuple{4, Int}[]
    dealt_dealer = Int64[]
    dealt_pone = Int64[]
    count_dealer = Int64[]
    count_pone = Int64[]
    regret_dealer = Float64[]
    regret_pone = Float64[]
    profile_dealer = Float64[]
    profile_pone = Float64[]
    p_play_dealer = Float64[]
    p_play_pone = Float64[]

    hRows = Dict{hType, UnitRange{Int}}()
    HRows = Dict{HType, Vector{Int}}()
    allh = hType[]
    hID = Dict{hType, Int}()
    allH = HType[]
    HID = Dict{HType, Int}()

    n = 0
    Hid = 1
    @showprogress 1 for (hid, h) in enumerate(keys(hCounts))

        push!(allh, h)
        hID[h] = hid

        D = getDiscards(h)
        nd = length(D)
        hRows[h] = (n+1:n+nd)

        for d in D
            push!(discard, d)
            ph = getPlayHand(h, d)
            push!(playhand, ph)
            H = counter(ph)
            if H in keys(HRows)
                push!(HRows[H], n+1)
            else
                HRows[H] = [n+1]
                push!(allH, H)
                HID[H] = Hid
                Hid += 1
            end

            push!(p_deal, hCounts[h] / N)
            push!(count_dealer, 0)
            push!(count_pone, 0)
            push!(dealt_dealer, 0)
            push!(dealt_pone, 0)
            push!(regret_dealer, 0.0)
            push!(regret_pone, 0.0)
            push!(profile_dealer, 1.0 / nd)
            push!(profile_pone, 1.0 / nd)
            push!(p_play_dealer, p_deal[end] * profile_dealer[end])
            push!(p_play_pone, p_deal[end] * profile_pone[end])

            n += 1

        end


    end

    df = DataFrame(p_deal = p_deal, discard = discard, playhand = playhand, 
                    count_dealer = count_dealer, count_pone = count_pone,
                    dealt_dealer = dealt_dealer, dealt_pone = dealt_pone,
                    regret_dealer = regret_dealer, regret_pone = regret_pone,
                    profile_dealer = profile_dealer, profile_pone = profile_pone,
                    p_play_dealer = p_play_dealer, p_play_pone = p_play_pone)


    Hprobs_dealer = Dict{HType, Float64}()
    Hprobs_pone = Dict{HType, Float64}()
    for H in allH
        Hprobs_dealer[H] = sum(p_play_dealer[HRows[H]])
        Hprobs_pone[H] = Hprobs_dealer[H]
    end

    return (df, hRows, HRows, allh, allH, hID, HID, Hprobs_dealer, Hprobs_pone)

end




# The matrix M stores the complete game tree for every pair of play hands. This is a lot of data, so it's highly optimized.
# The rows and columns of M are indexed with HID. The element at M[i,j] is a FlatTree, a space-efficient structure.
# To build M, we need to fill out each tree and then flatten it. We can build them one-by-one (well, in parallel), so we can
#   be less optimized, and work with a friendlier structure.
#
# A PlayState acts as both a state and a node in the game tree. Note its inefficient use os Vectors, etc. Later we'll extract
#   only the necessary data from these nodes.

mutable struct PlayState
    owner::Int64;
    hands::Vector{Vector{Int64}};
    history::Vector{Int64};
    diffs::Vector{Int64};
    total::Int64;
    pairlength::Int64;
    runlength::Int64;
    scores::Vector{Int64};
    children::Vector{PlayState};
    value::Int64;
    bestplay::Int64;
end
function Base.show(io::IO, ps::PlayState)
    print(io, "PlayState (p", ps.owner, "), values: ", [c.value for c in ps.children]);
end
function Base.show(io::IO, ::MIME"text/plain", ps::PlayState)
    print(io, "PlayState (p", ps.owner, ")\n");
    print(io, "  Hands:   ", ps.hands, "\n");
    print(io, "  History: ", ps.history, " (total: ", ps.total, ")\n");
    print(io, "  Scores:  ", ps.scores, "\n");
    print(io, "  Value:   ", ps.value, ", Best Play: ", ps.bestplay);
end
function showtree(ps::PlayState; depth = 0)
    display(" " ^ (depth) * string(ps))
    for c in ps.children
        showtree(c; depth = depth + 3);
    end
end


# Filling out a tree requires full play scoring, so scoreplay is defined next.

(@isdefined pSS) || (const pSS = (2, 6, 12));
(@isdefined run3diffs) || (const run3diffs = [[1, 1], [-1, -1], [2, -1], [-2, 1], [-1, 2], [1, -2]]);

"Get the value scored by the player making this play, and the new pair and run lengths."
function scoreplay(play::Int64, history::Vector{Int64}, diffs::Vector{Int64}, total::Int64,
                    pairlength::Int64, runlength::Int64)
    s = 0;
    if length(history) > 0
        if play == history[end]
            pairlength += 1;
            s += pSS[pairlength];
        else
            pairlength = 0;
        end
    else
        pairlength = 0;
    end
    if length(diffs) > 1
        if runlength == 0
            if diffs[(end-1):end] in run3diffs
                runlength += 1;
                s += runlength+2;
            else
                runlength = 0;
            end
        else
            currentrun = history[(end-1-runlength):end];
            if (play == minimum(currentrun) - 1) || (play == maximum(currentrun) + 1)
                runlength += 1;
                s += runlength+2;
            else
                runlength = 0;
            end
        end
    else
        runlength = 0;
    end
    (total == 15) && (s += 2);
    (total == 31) && (s += 1);
    return (s, pairlength, runlength);
end



"Recursively add children to node ps to build the complete game tree with ps as its root."
function solve!(ps::PlayState)

    if all(isempty.(ps.hands))
        newscores = copy(ps.scores);
        newscores[3-ps.owner] += 1;
        ps.value = newscores[1] - newscores[2];
        ps.bestplay = 0;
        return ps.value;
    end

    candidates = Int64[];
    childvalues = Int64[];

    for (ii, c) in enumerate(ps.hands[ps.owner])
        newtotal = cardvalues[c] + ps.total;
        (newtotal > 31) && (continue);
        (c in candidates) && (continue);
        push!(candidates, c);

        
        (length(ps.history) > 0) ? (newdiffs = vcat(ps.diffs, [c-ps.history[end]])) : (newdiffs = Int64[]);

        (s, newpairlength, newrunlength) = scoreplay(c, ps.history, newdiffs, newtotal,
                                                        ps.pairlength, ps.runlength);
        newscores = copy(ps.scores);
        newscores[ps.owner] += s;


        newhands = deepcopy(ps.hands); deleteat!(newhands[ps.owner], ii);

        push!(ps.children, PlayState(3 - ps.owner, newhands, vcat(ps.history, [c]), newdiffs, newtotal, 
                                    newpairlength, newrunlength, newscores, PlayState[], 0, 0));
        push!(childvalues, solve!(ps.children[end]));
    end

    if length(childvalues) > 0
        (ps.owner == 1) ? ((m, mi) = findmax(childvalues)) : ((m, mi) = findmin(childvalues));
        ps.value = m;
        ps.bestplay = candidates[mi];
    else
        if ps.history[end] == 0     # this is a double GO, so reset
            push!(ps.children,
                    PlayState(3 - ps.owner, deepcopy(ps.hands), vcat(ps.history, [0]), Int64[], 0, 
                    0, 0, ps.scores, PlayState[], 0, 0)
            );
        else                        # this is the first GO.
            newscores = copy(ps.scores); newscores[3 - ps.owner] += 1;    # other player gets a point
            push!(ps.children,
                    PlayState(3 - ps.owner, deepcopy(ps.hands), vcat(ps.history, [0]), ps.diffs, ps.total, 
                    ps.pairlength, ps.runlength, newscores, PlayState[], 0, 0)
            );
        end
        ps.value = solve!(ps.children[end]);
        ps.bestplay = 0;
    end

    return ps.value;

end


# The data model of the FlatTree is a flat list of the nodes of the game tree, in order of a breadth-first traversal.
# Each node has a value and a set of children. In practice, we are interested in the values of the children of the
#   node under consideration, so we model each node as a pair of tuples (vl...) and (ix...) where the vl are the 
#   childrens' values and the ix are the absolute indices of the child nodes within the flat tree.
# The values are stored as 8-bit integers and the indices at 16-bit integers.
#
# No two nodes ever have a child in common, so the number of nodes in the tree equals the length of either vector.
struct FlatTree
    child_indices::Tuple{Vararg{Union{NTuple{4, Int16}, NTuple{3, Int16}, NTuple{2, Int16}, NTuple{1, Int16}, Tuple{}}}}
    child_values::Tuple{Vararg{Union{NTuple{4, Int8}, NTuple{3, Int8}, NTuple{2, Int8}, NTuple{1, Int8}, Tuple{}}}}
end
Base.length(ft::FlatTree) = length(ft.child_indices)
Base.getindex(ft::FlatTree, ix::Int) = (ft.child_indices[ix], ft.child_values[ix])
Base.show(io::IO, ft::FlatTree) = "Flat Tree (" * string(length(ft)) * ")"


"Create a FlatTree from a tree rooted at the PlayState ps."
function flatten(ps::PlayState)

    I = Vector{Union{NTuple{4, Int16}, NTuple{3, Int16}, NTuple{2, Int16}, NTuple{1, Int16}, Tuple{}}}()
    V = Vector{Union{NTuple{4, Int8}, NTuple{3, Int8}, NTuple{2, Int8}, NTuple{1, Int8}, Tuple{}}}()

    Q = [ps]
    n = 1

    while !isempty(Q)
        node = popfirst!(Q)
        v = Int8[]
        i = Int16[]
        
        for child in node.children

            push!(v, child.value)
            # if the child is a leaf, then it's already represented by the above push. We don't need a new index
            #   too because there would be no values to put there anyway.
            if !isempty(child.children)         
                n += 1
                push!(i, n)
                push!(Q, child)
            end
        end
        push!(I, Tuple(i))
        push!(V, Tuple(v))
    end
    return FlatTree(Tuple(I), Tuple(V))
end




# This format is just fine for working in-memory. Unfortunately, it's pretty slow for serialization. packFlat goes one 
#   step further, concatenating all of the tuples and storing a separate vectore recording their lengths.
# This makes for efficient serialization and deserialization; the time gained more than offsets the extra processing
#   time lost converting between FlatTrees and FlatPacks. FlatPacks are *also* somewhat less space-efficient.
# NOTE: the index tuple and the values tuple will always have the same length, except when the index tuple is empty,
#   in which case the children under consideration are leaves, and thus the corresponding value tuples have length 1.
#   This will be important when *unpacking* FlatPacks. In this case, tuple_lengths[ix] will be zero.

struct FlatPack
    index_data::Tuple{Vararg{Int16}}
    value_data::Tuple{Vararg{Int8}}
    tuple_lengths::Tuple{Vararg{Int8}}
end

"Create a FlatPack from a FlatTree."
function packFlat(ft::FlatTree)
    index_data = Int16[]
    value_data = Int8[]
    tuple_lengths = Int8[]

    for ix in 1:length(ft)
        k = length(ft.child_indices[ix])
        push!(index_data, ft.child_indices[ix]...)
        push!(value_data, ft.child_values[ix]...)
        push!(tuple_lengths, k)
    end

    return FlatPack(Tuple(index_data), Tuple(value_data), Tuple(tuple_lengths))
end
# Not every pair of hands is possible, thus M will contain some copies of Nothing
packFlat(::Nothing) = Nothing


"Create a FlatTree from a FlatPack."
function unpackFlat(fp::FlatPack)
    child_indices = []
    child_values = []
    ix = 1
    for k in fp.tuple_lengths
        if k == 0
            push!(child_indices, ())
            push!(child_values, (fp.value_data[ix],))
            ix += 1
        else
            push!(child_indices, Tuple(fp.index_data[ix:(ix+k-1)]))
            push!(child_values, Tuple(fp.value_data[ix:(ix+k-1)]))
            ix += k
        end
    end
    return FlatTree(Tuple(child_indices), Tuple(child_values))
end
# Not every pair of hands is possible, thus M will contain some copies of Nothing
unpackFlat(::Nothing) = Nothing




"Build the matrix M of all possible game trees, in FlatTree format."
function buildM(allH, HID)

    nH = length(allH)
    M = Matrix{Union{Nothing, FlatTree}}(nothing, nH, nH)

    @showprogress 1 for H1 in allH
        @floop for H2 in allH

            any(values(merge(H1, H2)) .> 4) && continue

            i1 = HID[H1]
            i2 = HID[H2]

            ps = PlayState(2, [c2v(H1), c2v(H2)], Int64[], Int64[], 0, 0, 0, [0,0], PlayState[], 0, 0)
            solve!(ps)
            M[i1, i2] = flatten(ps)

        end
    end
    
    return M
end



"Save M to disk."
function saveM()
    serialize("data/M.jls", Folds.map(packFlat, M))
end

"Load M from disk."
function loadM()
    return Folds.map(unpackFlat, deserialize("data/M.jls"))
end













