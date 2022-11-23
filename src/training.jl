"""
  training.jl contains the structures and algorithms that underly the CFR loop. This includes information modeling
and the bayesian inference update step. There are some important constants defined here for efficiency as well.
"""

using DataStructures





## BIG NOTE: The include and exclude masks and methods assume all four suits exist in the deck. 
## IF YOU TEST ON A DECK WITH FEWER THAN 4 SUITS, YOU MUST CHANGE THE MAGIC NUMBER 4 THAT OCCURS 23 LINES DOWN.

"Generate a bit mask of size n_H x 4 x 13. imask[m,n,r] is true if HID^-1[m] contains at least n copies of rank r."
function generateIncludeMask(n_H, allH, HID)
    imask = falses(n_H, 4, 13)
    for r in 1:13
        for k in 1:4
            for H in allH
                Hid = HID[H]
                (H[r] >= k) && (imask[Hid, k, r] = true)
            end
        end
    end
    return imask
end

"Generate a bit mask of size n_H x 4 x 13. emask[m,n,r] is true if HID^-1[m] contains at most 4-n copies of rank r."
function generateExcludeMask(n_H, allH, HID)
    emask = falses(n_H, 4, 13)
    for r in 1:13
        for k in 1:4
            for H in allH
                Hid = HID[H]
                (H[r] <= (4-k)) && (emask[Hid, k, r] = true)
            end
        end
    end
    return emask
end



####################
## PLAY UTILITIES ##
####################



#   For development, and potentially for training, too, we define an information model that does not contain enough
# to support the bayesian update step. When using this model, probabilities will always be based on the base assumption
# of the current strategy profile, rather than updating in response to opponent plays.
#
#   I suspect that this version trains almost as well. It will be an interesting question to investigate whether
# there is a meaninful difference between the two approaches. Both will converge, so the questions are 1) do they converge
# to the same ϵ-solution? and 2) is the computational effeciency tradeoff worth it?


struct InformationModel_Base
    include::HType              # counts of the ranks known to be in the unknown hand.
    exclude::HType              # counts of the ranks known not to be in the unknown hand.
    model::BitVector            # Indices correspond to HID. model[ix] is True if HID^-1[ix] is admissable.
    probs::Vector{Float64}      # The probabilities I assign to each admissible hand
    pointers::Vector{Int64}     # Pointers into the FlatTrees corresponding to the hands in the model. These represent the 
                                #   game state we would be in if opponent has the corresponding hand H
    trees::Vector{Union{FlatTree, Nothing}}     # The row or column from M that corresponds to our hand
end

function InformationModel_Base(db::DB)
    return InformationModel_Base(
                Accumulator{Int64, Int64}(),
                Accumulator{Int64, Int64}(),
                trues(db.n_H),
                fill(1.0 / db.n_H, db.n_H),
                fill(1, db.n_H),
                fill(FlatTree(), db.n_H)
                )
end

function Base.show(io::IO, IM::InformationModel_Base)
    print(io, "IM_Base. Include: ", IM.include, ";  Exclude: ", IM.exclude, "\n")
end


#   A variation of Bayes' Theorem goes: P(B|A) = [ P(A|B)P(B) ] / [ P(A|B)P(B) + P(A|~B)P(~B) ]. Here, B is "opponent makes
# the play they just made", and A is "opponent has a particular hand H". P(B) is the prior; to evaluate P(A|B) we need an 
# approximation of opponent's knowledge, hence the addition of a metamodel to the information model.


struct InformationModel_Bayes
    include::HType              # counts of the ranks known to be in the unknown hand.
    exclude::HType              # counts of the ranks known not to be in the unknown hand.
    meta_include::HType         # counts of the ranks known by my opponent to be in my hand.
    meta_exclude::HType         # counts of the ranks known by my opponent not to be in my hand.
    model::BitVector            # Indices correspond to HID. model[ix] is True if HID^-1[ix] is admissable.
    metamodel::BitMatrix        # Both axes are indexed like HID. metamodel[m,n] is True means: If opponent has hand HID^-1[m],
                                #   then based on my knowledge, they should consider HID^-1[n] admissible. This only makes sense
                                #   for rows of metamodel that correspond to hands admissible according to my model.
                                # Note that for the row of metamodel corresponding to opponent's *actual* hand, my metamodel
                                #   will not be equal model, because opponent has different info than I do. All that's guaranteed
                                #   is that my metamodel row will always be True at the column corresponding to my actual hand
                                #   and opponent's model will be True at the row corresponding to my actual hand.
    p_model::Vector{Float64}    # The probabilities I assign to each admissible hand
    p_metamodel::Vector{Float64}    # The probabilities opponent assigns to each possible hand, according to the metamodel
    pointers_model::Vector{Int64}   # Pointers into the FlatTrees corresponding to the hands in the model. These represent the 
                                    #   game state we would be in if opponent has the corresponding hand H
    pointers_metamodel::Matrix{Int64}   # Pointers into the FlatTrees corresponding to the hands in the metamodel.
end

# TODO: duplicate the constructor, reset, etc. for the bayes version


## Functions for updating and resetting Information Models

function reset!(IM::InformationModel_Base, n_H::Int64)
    for k in 1:13
        IM.include[k] = 0
        IM.exclude[k] = 0
    end
    fill!(IM.model, true)
    fill!(IM.pointers, 1)
    fill!(IM.probs, 1.0/n_H)
end


function init_model!(IM::InformationModel_Base, knownRanks::HType, whichplayer::Int64, H::HType, db::DB)
    for (r, m) in knownRanks
        newExclude!(IM, r, m, db)
    end
    for ix in eachindex(IM.model)
        if whichplayer == 1
            (db.M[db.HID[H], ix] == Nothing) && (IM.model[ix] = false)
        elseif whichplayer == 2
            (db.M[ix, db.HID[H]] == Nothing) && (IM.model[ix] = false)
        end
    end
end

#NOTE: The following assumes that allH is ordered according to HID, so that HID[allH] == 1:n_H
"Set the probabilities in an information model according to the strategy profile corresponding to Hprobs."
function init_probs!(IM::InformationModel_Base, allH::Vector{HType}, Hprobs::Dict{HType, Float64})
    for ix in eachindex(IM.probs)
        IM.probs[ix] = Hprobs[allH[ix]]
    end
end

"Initialize IM.trees to the correct FlatTrees for our hand H. If whichplayer == 1, then we are dealer."
function init_trees!(IM::InformationModel_Base, whichplayer::Int64, H::HType, db)
    for ix in eachindex(IM.trees)
        if whichplayer == 1
            IM.trees[ix] = db.M[db.HID[H], ix]
        elseif whichplayer == 2
            IM.trees[ix] = db.M[ix, db.HID[H]]
        end
    end
end



"Initialize IM given known cards. knownRanks *includes the cards in H, the discard, and the turn*."
function init!(IM::InformationModel_Base, H::HType, knownRanks::HType, whichplayer::Int64, db::DB)
    reset!(IM, db.n_H)
    init_model!(IM, knownRanks, whichplayer, H, db)
    if whichplayer == 1
        init_probs!(IM, db.allH, db.Hprobs_pone)
    elseif whichplayer == 2
        init_probs!(IM, db.allH, db.Hprobs_dealer)
    end
    renormalize!(IM)
    init_trees!(IM, whichplayer, H, db)
end


function renormalize!(IM::InformationModel_Base)
    IM.probs ./= sum(@view IM.probs[IM.model])
end


"Update the IM's model based on the knowledge that mult new copies of r must be included in opponent's hand."
function newInclude!(IM::InformationModel_Base, r::Int64, mult::Int64, db::DB)
    IM.include[r] += mult
    IM.model .&= @view db.imask[:,IM.include[r],r]         # imask is a global defined in CFRibbage.jl
end


"Update the information model IM based on the knowledge that mult new copies of r must be excluded from opponent's hand."
function newExclude!(IM::InformationModel_Base, r::Int64, mult::Int64, db::DB)
    IM.exclude[r] += mult
    IM.exclude[r] = min(4-IM.include[r], IM.exclude[r])
    if IM.exclude[r] > 0
        IM.model .&= @view db.emask[:,IM.exclude[r], r]         # emask is a global defined in CFRibbage.jl
    end
end


"""
Update the information model IM in response to an opponent's play when total was playtotal.
This involves: 1) updating model; 2) updating pointers; 3) updating probs.
"""
function opponentPlay!(IM::InformationModel_Base, play::Int64, playtotal::Int64, db::DB)

    # Update model
    if play == 0   # GO
        for r in 1:min(13, (31 - playtotal))
            (IM.exclude[r] < 4) && newExclude!(IM, r, 4 - IM.exclude[r], db)
        end
    else            # a card was played
        newInclude!(IM, play, 1, db)
    end

    # Update pointers
    for (ix, tree) in enumerate(IM.trees)
        isnothing(tree) && continue             # no tree to point to
        (IM.model[ix] == false) && continue     # dead pointer
        tree[IM.pointers[ix]].isLeaf && continue

        for (jx, r) in enumerate(tree[IM.pointers[ix]].plays)      # look thru the potential plays for opp at this node
            if r == play
                IM.pointers[ix] = tree[IM.pointers[ix]].fi + jx - 1  # once we find the play that happened, jump
                break
            end
        end
    end

    # Update probs
    renormalize!(IM)

end


# TODO: refactor and pull out the for loop common to opponentPlay! and myPlay!
"Update the IM (base type) after I play. I learn nothing, and there's no metamodel, so just update pointers."
function myPlay!(IM::InformationModel_Base, play::Int64)
    for (ix, tree) in enumerate(IM.trees)
        isnothing(tree) && continue
        (IM.model[ix] == false) && continue
        tree[IM.pointers[ix]].isLeaf && continue

        for (jx, r) in enumerate(tree[IM.pointers[ix]].plays)
            if r == play
                IM.pointers[ix] = tree[IM.pointers[ix]].fi + jx - 1
                break
            end
        end
    end
end


"Choose the best play based on IM, as player whichplayer."
function getPlay(IM::InformationModel_Base, whichplayer::Int64)
    (whichplayer == 1) ? (whichmod = 1) : (whichmod = -1)       # p2 prefers negatives, but we want to compare positives.
    totalEVs = Accumulator{Int8, Float64}()
    for ix in eachindex(IM.model)
        IM.model[ix] || continue        # only consider live options
        # (IM.trees[ix][IM.pointers[ix]].isLeaf) && (return 0)
        (IM.trees[ix][IM.pointers[ix]].n == 1) && (return IM.trees[ix][IM.pointers[ix]].plays[1])
        for jx in 1:IM.trees[ix][IM.pointers[ix]].n
            totalEVs[IM.trees[ix][IM.pointers[ix]].plays[jx]] += whichmod * IM.probs[ix] * IM.trees[ix][IM.pointers[ix]].values[jx]
        end
    end
    bestEV = -999
    bestplay = -1
    for cand in keys(totalEVs)
        if totalEVs[cand] > bestEV
            bestEV = totalEVs[cand]
            bestplay = cand
        end
    end

    if bestplay == -1
        display("bad bestplay!")
    end

    return bestplay
end



"Get the result of one play hand. Assumes IM1 and IM2 have been initialized."
function playHand(H1::HType, H2::HType, IM1::InformationModel_Base, IM2::InformationModel_Base, db::DB)
    total = 0
    whoseturn = 2
    firstgo = false
    history = Int64[]

    # println("\nNEW HAND:")

    while (any(values(H1) .> 0) || any(values(H2) .> 0))
        
        # println(c2v(H1), " vs ", c2v(H2))
        # println("  history: ", history)    
        # println("  total: ", total)  
        # println("    p1 include: ", c2v(IM1.include), " and exclude: ", c2v(IM1.exclude))
        # println("    p2 include: ", c2v(IM2.include), " and exclude: ", c2v(IM2.exclude))


        if whoseturn == 1
            if any(values(H1) .> 0)
                p = Int64(getPlay(IM1, 1))
                myPlay!(IM1, p)
            else
                p = 0
            end
            push!(history, p)
            opponentPlay!(IM2, p, total, db)
            if p == 0
                if firstgo
                    total = 0
                    firstgo = false
                else
                    firstgo = true
                end
            else
                firstgo = false
                H1[p] -= 1
                total += cardvalues[p]
            end
        elseif whoseturn == 2
            if any(values(H2) .> 0)
                p = Int64(getPlay(IM2, 2))
                myPlay!(IM2, p)
            else
                p = 0
            end
            push!(history, p)
            opponentPlay!(IM1, p, total, db)
            if p == 0
                if firstgo
                    total = 0
                    firstgo = false
                else
                    firstgo = true
                end
            else
                firstgo = false
                H2[p] -= 1
                total += cardvalues[p]
            end

        end
  
        # println("  p", whoseturn, " plays ", p, ";  total now ", total)

        # flush(stdout)

        # sleep(0.1)

        whoseturn = 3-whoseturn

    end
    return history
end



"Given the two play hands and the history, calculate the net score."
function scoreHistory(H1, H2, history, db)
    ix = 1
    for r in history
        # println("current node: ", M[HID[H1], HID[H2]][ix], " at index ", ix)
        # println("  ", r, " played")


        jx = findfirst(db.M[db.HID[H1], db.HID[H2]][ix].plays .== r)
        if !db.M[db.HID[H1], db.HID[H2]][ix].isLeaf
            ix = db.M[db.HID[H1], db.HID[H2]][ix].fi + jx - 1
        else    
            return db.M[db.HID[H1], db.HID[H2]][ix].values[jx]
        end


    end
end





##############
## CFR GUTS ##
##############



# The core CFR loops is:
#   - Deal hands (we'll put this in CFRibbage.jl)
#   - Select discards according to current strategy
#   - Play and score selection and counterfactuals (my other potential discards against their actual discard)
#   - Calculate regrets and update strategy
# only step 3 has any nuance to it.
#
# The CFR loop function itself lives at the top level in CFRibbage.jl. The following are the functions that it calls.


"Get index of discard for hand h relative to rows from df."
function getDiscard(dfrows, whichplayer::Int64)
        x = rand()
    y = 0.0
    (whichplayer == 1) ? (profile = @view dfrows[:,:profile_dealer]) : (profile = @view dfrows[:,:profile_pone])
    for (ix, p) in enumerate(profile)
        y += p
        if x < y
            return ix
        end
    end
end



"Calculate counterfactual play scores (and the actual factual one, while we're at it)."
# function getPlayScores!(scorebuffer1::Vector{Int64}, scorebuffer2::Vector{Int64}, rows1, rows2, 
#                         di1::Int64, di2::Int64, known1::HType, known2::HType, turnrank::Int64,
#                         IM1::InformationModel_Base, IM2::InformationModel_Base, db::DB)

function getPlayScores!(scorebuffer1::Vector{Int64}, scorebuffer2::Vector{Int64}, rows1, rows2, 
                        di1::Int64, di2::Int64, known1::HType, known2::HType, turnrank::Int64,
                        IMs1::Vector{InformationModel_Base}, IMs2::Vector{InformationModel_Base}, db::DB)

    # score all of dealer's possible hands against pone's actual choice  

    @floop for (ix, row) in enumerate(eachrow(rows1))
        H1 = counter(row.playhand)
        H2 = counter(rows2[di2, :playhand])
        
        init!(IMs1[ix], H1, known1, 1, db)
        init!(IMs2[ix], H2, known2, 2, db)

        scorebuffer1[ix] = scoreHistory(H1, H2, playHand(copy(H1), copy(H2), IMs1[ix], IMs2[ix], db), db)
    end
        

    # ditto for pone
    @floop for (ix, row) in enumerate(eachrow(rows2))
        H1 = counter(rows1[di1, :playhand])
        H2 = counter(row.playhand)
        init!(IMs1[ix], H1, known1, 1, db)
        init!(IMs2[ix], H2, known2, 2, db)

        scorebuffer2[ix] = scoreHistory(H1, H2, playHand(copy(H1), copy(H2), IMs1[ix], IMs2[ix], db), db)

    end
    


end


"Get the show score for a hand (including the turn card). Set isCrib = true when scoring the crib."
function scoreHand(hand::handType, turncard::Card, isCrib = false)

    s = 0
    flushfound = false
    runlen = 0
    for combsize in 5:-1:2
        for comb in combinations(hand, combsize)

            if (combsize >= 4) && !flushfound       # check for a flush. If we find one, set flushfound to true
                if !isCrib || !(turncard in hand)   # the crib can't flush with the turn card.
                    if all([c.suit == comb[1].suit for c in comb])
                        s += combsize
                        flushfound = true
                        # println("  flush: ", comb)
                    end
                end
            end

            if (combsize >= 3) && (runlen <= combsize)  # check for a run. If we find one, set runlen = combsize. We'll still get other runs.
                ranks = sort([c.rank for c in comb])
                runfound = true
                for ix in 2:combsize
                    if ranks[ix] != ranks[ix-1] + 1
                        runfound = false
                        break
                    end
                end
                if runfound
                    runlen = combsize
                    s += combsize
                    # println("  run: ", comb)
                end
            end
            
            if sum([cardvalues[c.rank] for c in comb]) == 15        # check for 15
                s += 2
                # println("  fifteen: ", comb)
            end

            if combsize == 2                                        # check for a pair
                if comb[1].rank == comb[2].rank
                    s += 2
                    # println("  pair: ", comb)
                end
            end

        end
    end
    return s
end


"Find the hand to be scored in the show, given canonical form of the dealt hand and discard, plus the suit perm, and the turn card."
function getShowHand(h::hType, d::dType, turncard::Card, sperm)
    hand = [turncard]
    for ix in 1:4       # sperm has four entries, even if the hand has fewer suits
        for r in h[ix]
            (r in d[ix]) || push!(hand, Card(r, sperm[ix]))
        end
    end
    return hand
end

"Build the crib from each discard in canonical form."
function buildCrib(d1::dType, d2::dType, sp1, sp2, turncard::Card)
    crib = [turncard]
    for ix in 1:4
        for r in d1[ix]
            push!(crib, Card(r, sp1[ix]))
        end
        for r in d2[ix]
            push!(crib, Card(r, sp2[ix]))
        end
    end
    return crib
end



"Calculate counterfactual show scores (and the factual one too). Scores are net to dealer."
function getShowScores!(scorebuffer1::Vector{Int64}, scorebuffer2::Vector{Int64}, 
                        rows1, rows2, h1::hType, sp1, h2::hType, sp2, di1::Int64, di2::Int64, turncard::Card)

    # println("    showscores")

    # score all of dealer's possible hands against pone's actual choice                    
    for (ix, row) in enumerate(eachrow(rows1)) 
        scorebuffer1[ix] = scoreHand(getShowHand(h1, row.discard, turncard, sp1), turncard)
    end

    # scores are always net dealer, so pone scores are subtracted.
    for (ix, row) in enumerate(eachrow(rows2))
        scorebuffer2[ix] -= scoreHand(getShowHand(h2, row.discard, turncard, sp2), turncard)
    end

    # adjust scores according to opponent's actual show score (did i mention these are net to the dealer?)
    s1 = scorebuffer1[di1]
    scorebuffer1 .+= scorebuffer2[di2]
    scorebuffer2 .+= s1


    # now add cribs. They always count for the dealer.
    for (ix, row) in enumerate(eachrow(rows1))
        scorebuffer1[ix] += scoreHand(buildCrib(row.discard, rows2.discard[di2], sp1, sp2, turncard), turncard, true)
    end

    for (ix, row) in enumerate(eachrow(rows2))
        scorebuffer2[ix] += scoreHand(buildCrib(row.discard, rows1.discard[di1], sp2, sp1, turncard), turncard, true)
    end

end








