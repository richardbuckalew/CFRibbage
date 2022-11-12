struct PlayRecord
    dealerhand::handType
    poneHand::handType
    dealerdiscardindex::Int64
    ponediscardindex::Int64
    dealerdelta::Float64
    ponedelta::Float64
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