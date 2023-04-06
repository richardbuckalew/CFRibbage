

CSV.write("data/export/db.csv", CFRibbage.db.df[:, [:p_deal, :discard, :profile_dealer, :profile_pone, :p_play_dealer, :p_play_pone]],
            delim = ";")

# open("data/export/handIndex", "w") do f
#     JSON.print(f, db.hRows)
# end


