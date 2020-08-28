import plotly.graph_objects as go
import plotly.express as px

def visualize_2d_data(X):
    fig = px.scatter(x=X[:,0], y=X[:,1], width=600, height=600)
    fig.show()
    
def visualize_3d_data(X):
    fig = px.scatter_3d(x=X[:,0], y=X[:,1], z=X[:,2], width=600, height=600)
    fig.update_traces(marker=dict(size=3))
    fig.show()

def plot_2d_diffusion_embedding(x1, x2):
    fig = px.scatter(x=x1, y=x2)
    fig.show()
    
def plot_3d_diffusion_embedding(x1, x2, x3):
    fig = px.scatter_3d(x=x1, y=x2, z=x3)
    fig.show()
    
def display_eigenvector(E, eig_number):
    fig = px.scatter(x=range(len(E[:,eig_number])),
                     y=E[:,eig_number],
                     width=600, height=600)
    fig.show()