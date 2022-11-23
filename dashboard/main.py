from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
import os
import numpy as np
import datetime


data_path = os.path.dirname(__file__) + '/../data/'
TData = []
SStats = []
total_dealt = 0
total_time = 0
nps = 0



def load_data():
    global TData
    global SStats
    n_already = len(TData)
    filenames = [fn for fn in os.listdir(data_path) if (fn[:6] == 'sStats' or fn[:5] == 'tData')]
    N = []
    for fn in filenames:
        root = os.path.splitext(fn)[0]
        n = int(root.split('_')[1])
        if (n > n_already) and (not n in N):
            N.append(n)
    N.sort()
    for n in N:
        TData.append(json.load(open(os.path.join(data_path, 'tData_'+str(n)+'.json'))))
        SStats.append(json.load(open(os.path.join(data_path, 'sStats_'+str(n)+'.json'))))
    # TData = [json.load(open(os.path.join(data_path, 'tData_'+str(n)+'.json'))) for n in N]
    # SStats = [json.load(open(os.path.join(data_path, 'sStats_'+str(n)+'.json'))) for n in N]


def process_tData(tol = 1e-6):
    global total_dealt
    global total_time
    global nps

    quads = []
    nonzeros = []
    zero_fracs = []
    X = []

    total_dealt = 0
    total_time = 0.0

    for (n, tData) in enumerate(TData):
        total_dealt += tData['n_dealt']
        X.append(total_dealt)
        total_time += tData['dt']
        nonzeros.append([x for x in tData['deltas_dealer'] if x > tol] + [x for x in tData['deltas_pone'] if x > tol])
        zero_fracs.append( (len([x for x in tData['deltas_dealer'] if x < tol]) + len([x for x in tData['deltas_pone'] if x < tol])) / (len(tData['deltas_dealer']) + len(tData['deltas_pone'])) )
        q = []
        q.append(min(nonzeros[-1]))

        for p in [0.025, 0.5, 0.975]:
            q.append(np.quantile(nonzeros[-1], p))

        q.append(max(nonzeros[-1]))
        quads.append(q)

    nps = total_dealt / total_time

    return (X, quads, zero_fracs)



def make_cfig():
    f = make_subplots(specs=[[{'secondary_y':True}]])
    load_data()
    (x, quads, zero_fracs) = process_tData()

    # x = list(range(1, len(quads)+1))
    # print(len(x))
    # print(x)

    f.add_scatter(x = x, y = zero_fracs, fill='tonexty', line={'color':'lightgrey'})

    f.add_scatter(x = x, y = [q[0] for q in quads], line = {'color':'black', 'width':0.5, 'dash':'dot' }, secondary_y= True)
    f.add_scatter(x = x, y = [q[1] for q in quads], line = {'color':'black', 'width':1}, secondary_y= True)
    f.add_scatter(x = x, y = [q[2] for q in quads], line = {'color':'black', 'width':2}, fill = 'tonexty', secondary_y= True)
    f.add_scatter(x = x, y = [q[3] for q in quads], line = {'color':'black', 'width':1}, fill = 'tonexty', secondary_y= True)
    f.add_scatter(x = x, y = [q[4] for q in quads], line = {'color':'black', 'width':0.5, 'dash':'dot' }, secondary_y= True)

    f.update_layout(showlegend = False)
    f.update_layout(yaxis2={'type':'log'})
    f.update_layout(title = '{:,} hands dealt in {} ({:.1f} hands per second)'.format(total_dealt, datetime.timedelta(seconds=round(total_time)), nps))
    
    return f


def make_coverage(n):
    s = SStats[n]

    hcounts = s['hand_counts_dealer']
    for (k, v) in s['hand_counts_pone'].items():
        if k in hcounts.keys():
            hcounts[k] += v
        else:
            hcounts[k] = v

    f = go.Figure()
    X = list(hcounts.keys())
    X.sort()
    Y = [hcounts[x] for x in X]
    f.add_trace(go.Bar(x = X, y=Y))

    f.update_layout(title='Count of hands which have been dealt x times')
    return f


def make_discards(n):
    s = SStats[n]

    counts = [0 for i in range(15)]
    for (k, v) in s['active_discards_dealer'].items():
        counts[int(k)-1] += v
    for (k, v) in s['active_discards_pone'].items():
        counts[int(k)-1] += v
    # counts = [c / sum(counts) for c in counts]

    f = go.Figure()
    f.add_trace(go.Bar(x=list(range(1,16)), y=counts))

    f.update_layout(yaxis={'type':'log'})
    f.update_layout(title='Count of hands with x active discards')
    return f
    


def make_probs(n):
    s = SStats[n]
    counts = s['HpHist_dealer'][1]
    for (ix, c) in enumerate(s['HpHist_pone'][1]):
        counts[ix] += c
    # counts = [c / sum(counts) for c in counts]

    f = go.Figure()
    f.add_trace(go.Bar(x = s['HpHist_dealer'][0], y=counts))

    f.update_layout(yaxis={'type':'log'})
    f.update_layout(title='Distribution of play hand probabilities')
    return f





app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(id='convergence', responsive = True)
    ], style = {'width':'49%', 'display':'inline-block'}),
    html.Div([
        dcc.Graph(id='coverage', responsive = True)
    ], style = {'width':'49%', 'display':'inline-block'}),
    html.Div([
        dcc.Graph(id='hprobs', responsive = True)
    ], style = {'width':'49%', 'display':'inline-block'}),
    html.Div([
        dcc.Graph(id='discards', responsive = True)
    ], style = {'width':'49%', 'display':'inline-block'}),
    dcc.Interval(id='iv', interval=600000, n_intervals = 0),
])


@app.callback(
    Output('coverage', 'figure'),
    Output('discards', 'figure'),
    Output('hprobs', 'figure'),
    Input('convergence', 'clickData')
    )
def update_discards(cd):
    if cd is None:
        return (go.Figure(), go.Figure(), go.Figure())

    n = cd['points'][0]['pointNumber']

    f0 = make_coverage(n)
    f1 = make_discards(n)
    f2 = make_probs(n)
    return (f0, f1, f2)



@app.callback(
    Output('convergence', 'figure'),
    Input('iv', 'n_intervals')
)
def refresh_data(n):
    f1 = make_cfig()
    return f1




if __name__ == '__main__':
    app.run_server(debug = True)














