from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from model_util import pre_trained_models, dog_names
from util import get_metric_graphs, model_preds

models = [model for model in pre_trained_models]

_header = html.Header([
    html.H1('Dog breed classification with convolutional neural networks trained with transfer learning'),
    html.Br(),
    html.A('Project details here', href='https://github.com/jimtoledo/Udacity-Dog-Project', target='_blank'),
    html.Hr()
])

model_tab = dbc.Tab(label='Demo models', tab_id='model_tab', active_tab_class_name='fw-bold', children=[
    html.Br(),
    html.H4('1. Upload images'),
    html.Hr(),
    dcc.Upload(
        id='upload-image',
        accept='image/png, image/jpeg',
        children = html.Div(['Drag and Drop or ', html.A('Select Files')]),
        multiple = True,
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
    ),
    dbc.Container(id='image-output', style={'max-height': '50vh', 'max-width': '100vw', 'overflow' : 'auto'}, fluid=True),
    html.Br(),

    html.H4('2. Select Model'),
    html.Hr(),
    dcc.Dropdown(options=models, value=models[0], searchable=False, clearable=False, id='select-model'),
    html.Br(),

    html.H4('3. Run the model'),
    html.Hr(),
    dbc.Button('Run', id='run-model', size="lg", disabled=True),
    html.Hr(),
    dbc.Spinner(dbc.Container(id='results', class_name='dbc'))
])

data_tab = dbc.Tab(label='Model stats', tab_id='stats_tab', active_tab_class_name='fw-bold', children=[
    html.Br(),
    html.H4('Model train/validate/test loss and accuracy'),
    html.Hr(),
    html.Img(src=get_metric_graphs(), style={'max-width': '100%'}),
    html.Br(),
    html.H4('Explore model accuracy on testing set'),
    html.Hr(),
    dbc.Container([
        dbc.Row([
            dbc.Col([dbc.Button('Select All', id='dog-all', class_name='me-3'),dbc.Button('Clear All', id='dog-clear', class_name='me-3')]),
            dbc.Col()
        ]),
        dbc.Row(html.Br()),
        dbc.Row([
                dbc.Col(dcc.Checklist(dog_names, dog_names, id='dog-filter', className='me-3', style={'max-height':'100vh','overflowY' : 'auto'}), width=3),
                dbc.Col([dbc.Row(dbc.Container(dash_table.DataTable(model_preds.groupby('model')['accuracy'].mean().reset_index()[['model', 'accuracy']].to_dict('records'), id='summary-table'))),
                        dbc.Row(html.Br()),
                        dbc.Row(dbc.Container(dash_table.DataTable(model_preds.to_dict('records'), 
                                                                    columns=[{"name": i, "id": i} for i in model_preds.columns],
                                                                    style_data={'whiteSpace':'normal', 'height':'auto'},
                                                                    style_table={'width':'100%', 'overflowX':'visible'},
                                                                    filter_action="native", sort_action="native", sort_mode="multi", id='pred-table'),
                                                                    style={'max-height' : '85vh','overflowY' : 'auto'}))], width=9)
        ], style={'max-height' : '100vh', 'max-width' : '100vw'})
], style={'max-height':'100vh'})

    #dcc.Dropdown(dog_names, dog_names, multi=True)
])

layout = dbc.Container([
    html.Title(),
    _header,
    dbc.Tabs([
        model_tab,
        data_tab
    ])
], class_name='dbc')
