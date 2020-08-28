import plotly.graph_objects as go
import plotly.express as px

def plot_2d_diffusion_embedding(x1, x2):
    fig = px.scatter(x=x1, y=x2)
    fig.show(renderer='iframe')
    
def plot_3d_diffusion_embedding(x1, x2, x3):
    fig = px.scatter_3d(x=x1, y=x2, z=x3)
    fig.show(renderer='iframe')