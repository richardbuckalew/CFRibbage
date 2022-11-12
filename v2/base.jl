using Combinatorics, DataFrames, DataStructures, IterTools, ProgressMeter


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
#  - HID: A dict mapping accumulator hands H to integer indices. Used for efficiently indexing M.

"Build a new strategy dataframe, and all supporting structures, from a deck."
function buildDF(deck)

    hCounts = dealAllHands(deck)
    N = sum(values(hCounts))

    p_deal = Float64[]
    discard = dType[]
    playhand = NTuple{4, Int}[]
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
    allH = HType[]
    HID = Dict{HType, Int}()

    n = 0
    Hid = 1
    for h in keys(hCounts)

        push!(allh, h)

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
                    regret_dealer = regret_dealer, regret_pone = regret_pone,
                    profile_dealer = profile_dealer, profile_pone = profile_pone,
                    p_play_dealer = p_play_dealer, p_play_pone = p_play_pone)


    Hprobs_dealer = Dict{HType, Float64}()
    Hprobs_pone = Dict{HType, Float64}()
    for H in allH
        Hprobs_dealer[H] = sum(p_play_dealer[HRows[H]])
        Hprobs_pone[H] = Hprobs_dealer[H]
    end

    return (df, hRows, HRows, allh, allH, HID, Hprobs_dealer, Hprobs_pone)

end



