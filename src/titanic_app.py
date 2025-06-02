
import dash_bootstrap_components as dbc
from dash import Dash
from titanic_ml import ML


# Incorporate data
ml = ML()

layout = ml.run()

# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


# App layout
app.layout = dbc.Container(layout)

if __name__ == '__main__':
  app.run(debug=True, port=55003)