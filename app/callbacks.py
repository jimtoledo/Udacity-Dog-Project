from app import app
from dash import Input, Output, State
from dash import html
import dash_bootstrap_components as dbc
from model_util import predict_breed, dog_names
from util import model_preds

@app.callback(
    Output('image-output', 'children'),
    Output('run-model', 'disabled'),
    Input('upload-image', 'contents'),
    prevent_initial_call = True
)
#preview images that are uploaded, enable run button
def update_image(contents):
    children = [html.Img(src=content) for content in contents]
    return children, len(children) == 0

@app.callback(
    Output('results', 'children'),
    Input('run-model', 'n_clicks'),
    State('select-model', 'value'),
    State('image-output', 'children'),
    prevent_initial_call = True
)
#run the model and display the results
def run_model(_, model, images):
    rows = []
    cards = []
    for image in images:
        content = image['props']['src'] #get image content
        pred = predict_breed(content[content.rindex('base64,')+7:], model)

        card = dbc.Card([
                dbc.CardBody(html.H3(pred)),
                dbc.CardImg(src=content, bottom=True, style={"max-width" : "30vw", "max-height" : "35vh"})
            ],
            style= {"max-width" : "30vw", "max-height" : "40vh"},
            class_name='me-2'
        )
        cards.append(dbc.Col(card))
        if(len(cards) == 3):
            rows.append(dbc.Row(cards, style={'max-height':'50vh'}))
            cards = []

    if len(cards) > 0:
        for _ in range(3-len(cards)):
            cards.append(dbc.Col())
        rows.append(dbc.Row(cards))

    return rows

@app.callback(
    Output('summary-table', 'data'),
    Output('pred-table', 'data'),
    Input('dog-filter', 'value'),
    prevent_initial_call = True
)
#update data tables based on filter
def update_tables(value):
    df_preds = model_preds[model_preds['predicted'].isin(value)]
    summary_data = df_preds.groupby('model')['accuracy'].mean().reset_index()[['model', 'accuracy']].to_dict('records')
    pred_data = df_preds.to_dict('records')
    return summary_data, pred_data

@app.callback(
    Output('dog-filter', 'value', allow_duplicate=True),
    Input('dog-all', 'n_clicks'),
    prevent_initial_call = True
)
#select all dogs
def select_all(_):
    return dog_names

@app.callback(
    Output('dog-filter', 'value', allow_duplicate=True),
    Input('dog-clear', 'n_clicks'),
    prevent_initial_call = True
)
#clear all dogs
def clear_all(_):
    return []

