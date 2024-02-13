from dash import Dash
import dash_bootstrap_components as dbc
from layout import layout
from dash import Output, Input

app = Dash(__name__, title='Dog Breed Identification', external_stylesheets=[dbc.themes.JOURNAL])
app.layout = layout

from callbacks import *

if __name__ == '__main__':
    app.run(debug=True)