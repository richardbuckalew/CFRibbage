"""
analytics.jl holds the tools for logging progress and processing it.
  A web dashboard (probably plotly.dash; I'll consider Genie too) will monitor training progress. So logs
need to be threadsafe. To achieve this, we'll log regularly and have the progress monitor lag behind by one
full log file.

  The most obvious format for a log file is a snapshot of the current state. These are large,
not cross-platform (without work), and not suited for cloud storage, so we will also log summary statistics 
in json form.

  Training will proceed in segments; during a segment the results of dealt hands will be stored in a TrainingData
object and used to generate a SummaryStats object at the end of the segment. The real-time data stored includes:
  - IDs of hands dealt to each player (from hID)
  - id of the discard chosen by each player (integer index within hRows)
  - net scores achieved for each hand dealt
  - the (normed) change in strategy for each hand dealt to each player
"""


using JSON, StatsBase




struct TrainingData
    n_dealt::Int64                              # of hands in this training set
    dealt_dealer::Vector{Int64}                 # a list of ids from hID dealt to the dealer
    dealt_pone::Vector{Int64}
    discards_dealer::Vector{Int64}              # a list of the relative indices of each discard played
    discards_pone::Vector{Int64}
    scores::Vector{Int64}                       # the net score (value) of each hand played
    deltas::Vector{Float64}                     # the ∞-normed deltas of each hand
    IGs::Vector{Float64}                        # the LK-divergence from each hand update
end

struct SummaryStats
    n_hands::Int64                              # Total number of training hands dealt
    cpu_time::Float64                           # total time spent training
    coverage_dealer::Float64                    # Percentage of all hands seen
    coverage_pone::Float64                      
    hand_counts_dealer::Accumulator{Int64, Int64}      # keys: # of times a hand has been seen; values: number of such hands
    hand_counts_pone::Accumulator{Int64, Int64}
    active_discards_dealer::Accumulator{Int64, Int64}  # keys: # of active discards; values: # of such hands
    active_discards_pone::Accumulator{Int64, Int64}
    average_score::Float64                      # average net score of all hands so far
    delta_perhand::Float64                      # average magnitude of strategy change when a hand is dealt, per hand
    IG_perhand::Float64                         # average information gain per hand
    Hprob_max::Float64                          # the highest play hand probability
    HpHist_dealer::Tuple{Vector{Float64}, Vector{Int64}}    # histogram of Hprobs. Format: (bin_bounds, counts)
    HpHist_pone::Tuple{Vector{Float64}, Vector{Int64}}
end
function Base.show(io::IO, ::MIME"text/plain", S::SummaryStats)
    print(io, "Summary Stats for ", S.n_hands, " hands in ", S.cpu_time, "s:\n")
    print(io, "    Coverage: ", (round(S.coverage_dealer, digits=4), round(S.coverage_pone, digits = 4)), "\n")
    print(io, "    Average Δ: ", round(S.delta_perhand, digits=4), "\n")
end


"Create a SummaryStats object from a snapshot of df and related data."
function summarize(df, dt, hRows, Hprobs_dealer, Hprobs_pone, old_stats::SummaryStats, tData::TrainingData)

    # stats to be gleaned from df
    covered_dealer = 0 
    covered_pone = 0

    hand_counts_dealer = Accumulator{Int64, Int64}()
    hand_counts_pone = Accumulator{Int64, Int64}()

    active_discards_dealer = Accumulator{Int64, Int64}()
    active_discards_pone = Accumulator{Int64, Int64}()

    for rows in values(hRows)
        (df.dealt_dealer[rows.start] > 0) && (covered_dealer += 1)
        hand_counts_dealer[df.dealt_dealer[rows.start]] += 1
        ad_key = count(x -> (x > 1e-6), df.profile_dealer[rows])
        active_discards_dealer[ad_key] += 1

        (df.dealt_pone[rows.start] > 0) && (covered_pone += 1)
        hand_counts_pone[df.dealt_pone[rows.start]] += 1
        ad_key = count(x -> (x > 1e-6), df.profile_pone[rows])
        active_discards_pone[ad_key] += 1
    end

    coverage_dealer = covered_dealer / length(allh)
    coverage_pone = covered_pone / length(allh)


    # Hprobs
    Hprob_max = max(maximum(values(Hprobs_dealer)), maximum(values(Hprobs_pone)))
    edges = LinRange(0.0, Hprob_max, 101)
    hpd = fit(Histogram, collect(values(Hprobs_dealer)), edges)
    hpp = fit(Histogram, collect(values(Hprobs_pone)), edges)
    HpHist_dealer = (edges, hpd.weights)
    HpHist_pone = (edges, hpp.weights)


    # Training Data
    n_hands = old_stats.n_hands + tData.n_dealt
    delta_perhand = (old_stats.n_hands * old_stats.delta_perhand + sum(tData.deltas)) / n_hands
    IG_perhand = (old_stats.n_nhands * old_stats.IG_perhand + sum(tData.IGs)) / n_hands
    average_score = (old_stats.n_hands * old_stats.average_score + sum(tData.scores)) / n_hands


    # Old stats
    cpu_time = old_stats.cpu_time + dt



    return SummaryStats(n_hands, cpu_time, coverage_dealer, coverage_pone, hand_counts_dealer, hand_counts_pone,
                         active_discards_dealer, active_discards_pone, average_score, delta_perhand, IG_perhand,
                         Hprob_max, HpHist_dealer, HpHist_pone)

end




