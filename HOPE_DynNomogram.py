import numpy as np
import dash
from dash import dcc, html, Output, Input
import plotly.graph_objs as go

# --- 1 ---
features = ['DL risk score_std_cat', 'ALBI_risk_score_std_cat', 'MVI_risk_score_std_cat']

coefs = [0.6169, 0.2703, 0.9427] 
feature_ranges = [(0, 1), (0, 1), (0, 1)]
feature_labels = [('0', '1'), ('0', '1'), ('0', '1')]


BASELINE_SURVIVAL_MAP = {
    12: 0.8690447,  
    36: 0.6679847,  
    60: 0.3763885   
}

# --- 2 ---

contribs = np.abs(np.array(coefs) * (np.array([r[1] - r[0] for r in feature_ranges])))
max_contrib = max(contribs)
score_scale = 100 / max_contrib
total_score_max = sum([abs(c * (r[1]-r[0])) * score_scale for c, r in zip(coefs, feature_ranges)])

def calc_points(values):
    points = []
    for coef, v, r in zip(coefs, values, feature_ranges):
        if coef >= 0:
            p = coef * (v - r[0]) * score_scale
        else:
            p = coef * -(v - r[0]) * score_scale
        points.append(p)
    total = sum(points)
    return points, total

def calc_survival_prob(total_points, baseline_survival):
    lp = total_points / score_scale
    hazard_ratio = np.exp(lp)
    prob = baseline_survival ** hazard_ratio
    return prob

def get_prob_scale(baseline_survival):
    probs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    valid_probs = [p for p in probs if p <= baseline_survival]
    
    prob_scores = []
    for p in valid_probs:
        try:
            val = np.log(p) / np.log(baseline_survival)
            
            # The calculation is correct: total points = score_scale * LP
            lp = np.log(val) 
            points = lp * score_scale 
            prob_scores.append(points)
        except:
            prob_scores.append(-999)
            
    return valid_probs, prob_scores

# --- 3 ---

app = dash.Dash(__name__)

def feature_input(i):
    labels = feature_labels[i]
    return html.Div([
        html.Label(features[i], style={'font-weight':'bold', 'color': '#333'}), 
        dcc.RadioItems(
            id=f'feat-{i}',
            options=[
                {'label': f" {labels[0]} ", 'value': 0},
                {'label': f" {labels[1]} ", 'value': 1}
            ],
            value=0,
            inline=True,
            labelStyle={'display': 'inline-block', 'margin-left': '10px', 'margin-right': '10px', 'color': '#555'}
        )
    ], style={'marginRight': '30px', 'display':'inline-block', 'font-size':'18px'})

app.layout = html.Div([
    html.Div([
        
        html.Div(
            html.H2('HOPE Nomogram: Dynamic Survival Prediction', style={
                'textAlign': 'center', 
                'margin': 0,
                'color': 'white'
            }),
            style={
                'background': '#1a5f7a', 
                'borderTopLeftRadius': '16px',
                'borderTopRightRadius': '16px',
                'padding': '18px 0 10px 0',
                'borderBottom': '1px solid #0f4c63'
            }
        ),
        
       
        html.Div([
            html.Label("Select Prediction Time Point:", style={'font-weight':'bold', 'marginRight': '20px', 'color': '#333'}),
            dcc.RadioItems(
                id='time-selector',
                options=[
                    {'label': '1 Year (12M)', 'value': 12},
                    {'label': '3 Years (36M)', 'value': 36},
                    {'label': '5 Years (60M)', 'value': 60}
                ],
                value=36, 
                inline=True,
                labelStyle={'display': 'inline-block', 'margin-left': '20px', 'margin-right': '20px', 'cursor': 'pointer', 'color': '#555'}
            )
        ], style={'padding': '14px 0', 'textAlign': 'center', 'background': '#f0f4f7'}), 

       
        html.Div(
            [feature_input(i) for i in range(len(features))],
            style={
                'background': '#ffffff', 
                'padding': '22px 0 18px 0',
                'textAlign': 'center',
                'borderBottom': '1px solid #ddd'
            }
        ),
        
        
        dcc.Graph(id='nomogram-graph', config={'displayModeBar': False}),
        
    ],style={
        'maxWidth': '1100px',
        'margin': '40px auto',
        'background': '#fff',
        'border': '1px solid #d0d0d0', 
        'borderRadius': '18px',
        'boxShadow': '0 10px 30px 0 rgba(0,0,0,0.15)',
        'padding': '0 36px 24px 36px',
    })
])

# --- 4 ---

@app.callback(
    Output('nomogram-graph', 'figure'),
    [Input(f'feat-{i}', 'value') for i in range(len(features))] + 
    [Input('time-selector', 'value')] 
)
def update_nomogram(*args):
    time_point = args[-1]
    feat_values = list(args[:-1])
    
    baseline_survival = BASELINE_SURVIVAL_MAP.get(time_point)
    
    # Error handling and label setup
    if baseline_survival is None:
        baseline_survival = 0.6679847 
        time_point_label = "ERROR"
    
    if time_point == 12:
        time_point_label = "1-Year"
    elif time_point == 36:
        time_point_label = "3-Year"
    elif time_point == 60:
        time_point_label = "5-Year"
    else:
        time_point_label = f"{time_point} Months"
        
    points, total = calc_points(feat_values)
    prob = calc_survival_prob(total, baseline_survival) 
    
    fig = go.Figure()
    y0 = len(features) + 4
    ygap = 0.5
    x_axis_move = 35 
    FEATURE_NAME_X_POS = -30 

    # Plot Colors 
    FEATURE_MARKER_COLOR = '#1f77b4'
    TOTAL_POINTS_COLOR = '#d62728'  
    PROB_MARKER_COLOR = '#2ca02c'   

    # 1. Points  (0-100)
    fig.add_trace(go.Scatter(x=[0+x_axis_move, 100+x_axis_move], y=[y0, y0], mode='lines', line=dict(color='#333', width=2), showlegend=False))
    for pt in np.linspace(0, 100, 11):
        fig.add_trace(go.Scatter(x=[pt+x_axis_move, pt+x_axis_move], y=[y0, y0+0.1], mode='lines', line=dict(color='#333', width=2), showlegend=False))
        fig.add_trace(go.Scatter(x=[pt+x_axis_move], y=[y0+0.15], mode='text', text=[f'{int(pt)}'], textfont=dict(size=14, color='#333'), textposition='top center', showlegend=False))
    fig.add_trace(go.Scatter(x=[FEATURE_NAME_X_POS], y=[y0], mode='text', text=['Points'], textfont=dict(size=16, family='Arial', color='#333',), textposition='middle right', showlegend=False))

    # 2.
    for i, (feat, coef, r, labels) in enumerate(zip(features, coefs, feature_ranges, feature_labels)):
        y = y0 - (i+1)*ygap
        max_score = abs(coef * (r[1] - r[0])) * score_scale
        
        # 轴线
        fig.add_trace(go.Scatter(x=[0+x_axis_move, max_score+x_axis_move], y=[y, y], mode='lines', line=dict(color='#555', width=2), showlegend=False))
        # 刻度
        fig.add_trace(go.Scatter(x=[0+x_axis_move, 0+x_axis_move], y=[y, y-0.1], mode='lines', line=dict(color='#555', width=2), showlegend=False))
        fig.add_trace(go.Scatter(x=[max_score+x_axis_move, max_score+x_axis_move], y=[y, y-0.1], mode='lines', line=dict(color='#555', width=2), showlegend=False))

        left_label = labels[0] if coef >= 0 else labels[1]
        right_label = labels[1] if coef >= 0 else labels[0]

        fig.add_trace(go.Scatter(x=[0+x_axis_move], y=[y-0.15], mode='text', text=[left_label], textfont=dict(size=14, color='#555'), showlegend=False, textposition='bottom center'))
        fig.add_trace(go.Scatter(x=[max_score+x_axis_move], y=[y-0.15], mode='text', text=[right_label], textfont=dict(size=14, color='#555'), showlegend=False, textposition='bottom center'))

        val = feat_values[i]
        this_score = (coef * (val - r[0])) * score_scale if coef >= 0 else (coef * -(val - r[0])) * score_scale
            
        # Feature Marker
        fig.add_trace(go.Scatter(x=[this_score+x_axis_move], y=[y], mode='markers', marker=dict(size=14, color=FEATURE_MARKER_COLOR, line=dict(width=1, color='white')), showlegend=False))
        fig.add_trace(go.Scatter(x=[FEATURE_NAME_X_POS], y=[y], mode='text', text=[feat], textfont=dict(size=16, color='#333'), showlegend=False, textposition='middle right'))

    # 3. Total Points 
    y_total = y0 - (len(features)+1)*ygap
    # Total Points  0 to total_score_max 
    fig.add_trace(go.Scatter(x=[0+x_axis_move, total_score_max+x_axis_move], y=[y_total, y_total], mode='lines', line=dict(color='#1a5f7a', width=3), showlegend=False))
    fig.add_trace(go.Scatter(x=[FEATURE_NAME_X_POS], y=[y_total], mode='text', text=['Total Points'], textfont=dict(size=16, color='#333'), showlegend=False, textposition='middle right'))
    
    tick_step = 10 if total_score_max > 50 else 5
    for pt in np.arange(0, total_score_max + 1, tick_step):
         fig.add_trace(go.Scatter(x=[pt+x_axis_move, pt+x_axis_move], y=[y_total, y_total-0.1], mode='lines', line=dict(color='#555', width=2), showlegend=False))
         fig.add_trace(go.Scatter(x=[pt+x_axis_move], y=[y_total-0.15], mode='text', text=[int(pt)], textfont=dict(size=12, color='#555'), showlegend=False, textposition='bottom center'))

    # Total Points Marker
    fig.add_trace(go.Scatter(x=[total+x_axis_move], y=[y_total], mode='markers', marker=dict(size=14, color=TOTAL_POINTS_COLOR, line=dict(width=1, color='white')), showlegend=False))
    fig.add_trace(go.Scatter(x=[total+x_axis_move], y=[y_total+0.08], mode='text', text=[f'{total:.1f}'], textfont=dict(size=16, color=TOTAL_POINTS_COLOR), showlegend=False, textposition='top center'))

    # 4. 
    y_prob = y0 - (len(features)+2)*ygap
    probs, prob_scores = get_prob_scale(baseline_survival) 
    
    valid_map = [(s, p) for s, p in zip(prob_scores, probs)]
    
    if valid_map:
        
        # Survival Probability  0 to total_score_max
        fig.add_trace(go.Scatter(x=[0 + x_axis_move, total_score_max + x_axis_move], y=[y_prob, y_prob], mode='lines', line=dict(color='#1a5f7a', width=3), showlegend=False))
        
        for s, p in valid_map:
           
            if 0 <= s <= total_score_max:
                fig.add_trace(go.Scatter(x=[s+x_axis_move, s+x_axis_move], y=[y_prob-0.1, y_prob], mode='lines', line=dict(color='#555', width=2), showlegend=False))
                fig.add_trace(go.Scatter(x=[s+x_axis_move], y=[y_prob-0.15], mode='text', text=[f'{p:.2f}'], textfont=dict(size=14, color='#555'), showlegend=False, textposition='bottom center'))
    
    fig.add_trace(go.Scatter(x=[FEATURE_NAME_X_POS], y=[y_prob], mode='text', text=[f'{time_point_label} Survival Probability'], textfont=dict(size=16, color='#333'), showlegend=False, textposition='middle right'))

    # Probability Marker
    fig.add_trace(go.Scatter(x=[total+x_axis_move], y=[y_prob], mode='markers', marker=dict(size=14, color=PROB_MARKER_COLOR, line=dict(width=1, color='white')), showlegend=False))
    fig.add_trace(go.Scatter(x=[total+x_axis_move], y=[y_prob+0.08], mode='text', text=[f'{prob:.2%}'], textfont=dict(size=16, color=PROB_MARKER_COLOR), showlegend=False, textposition='top center'))


    max_marker_x = total_score_max + x_axis_move
    final_x_range_max = max_marker_x + 15 

    fig.update_layout(
        height=550, 
        yaxis=dict(showticklabels=False, range=[y_prob-1, y0+1], zeroline=False),
       
        xaxis=dict(showticklabels=False, range=[-40, final_x_range_max], zeroline=False), 
        margin=dict(l=0, r=20, t=20, b=20),
        plot_bgcolor='#fcfcfc',
        font=dict(family="Arial, sans-serif")
    )

    return fig

if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=8050)
