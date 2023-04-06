using JSON3
using Genie, Genie.Requests, Genie.Renderer.Json

function cards_to_json(hand)
    return JSON3.write([("A23456789TJQK")[c.rank] * ("SHDC")[c.suit] for c in hand])
end


function get_discard(db, hand, isdealer)
    h, sp = canonicalize(hand)
    rows = @view db.df[db.hRows[h], [:discard, :profile_dealer, :profile_pone]]
    P = isdealer ? cumsum(rows.profile_dealer) : cumsum(rows.profile_pone)
    r = rand()
    let ix
        for (i,p) in enumerate(P)
            if r < p
                ix = i
                break
            end
        end
        d = rows.discard[ix]
        D = [Card(rank, sp[suit]) for suit in 1:4 for rank in d[suit]]
        return D
    end
end


"""Query format: hand::handType and isdealer::Bool"""
function handle_discard_query(db, query)
    #qDict = JSON3.read(query)

    hand = Card[]
    for n in query["hand"]
        rank = findfirst(n[1], "A23456789TJQK")
        suit = findfirst(n[2], "SHDC")
        push!(hand, Card(rank, suit))
    end
    D = get_discard(db, hand, query["isdealer"])
    response = [("A23456789TJQK")[c.rank] * ("SHDC")[c.suit] for c in D]
    return JSON3.write(Dict("discard" => response))
end


route("/hello") do 
    return "Hi ROSIE"    
end

route("/discard", method = POST) do 
    query = jsonpayload()
    display(query)
    response = handle_discard_query(db, query)
    return response
end

up()

# CN = Dict("hand" => ["AS", "2C", "6H", "KD", "JS", "TS"], "isdealer" => false)
# Q = JSON3.write(CN)

# D = handle_discard_query(db, Q)
# HTTP.request("POST", "http://localhost:8000/discard", [("Content-Type", "application/json")], Q)