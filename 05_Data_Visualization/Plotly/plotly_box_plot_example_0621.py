#!/usr/bin/env python
# coding: utf-8

# In[131]:


# Install packages 
#!pip install --upgrade plotly notebook cufflinks


# In[132]:


# Import libraries
import pandas as pd
import numpy as np
import cufflinks as cf
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

# Initialize notebook mode (must be called after import)
init_notebook_mode(connected=True)
cf.go_offline()


# In[133]:


# Initialize Plotly and Cufflinks for Jupyter Notebook
init_notebook_mode(connected=True)  # Enable Plotly for offline use in notebooks
cf.go_offline()                     # Set cufflinks to offline mode
cf.set_config_file(offline=True, world_readable=True)  # Configure cufflinks settings


# In[134]:


# Check versions (optional, useful for debugging)
import plotly
print("Plotly version:", plotly.__version__)
print("Cufflinks version:", cf.__version__)


# In[135]:


cf.go_offline()


# In[136]:


# Create a DataFrame with 100 rows and 4 columns filled with random numbers
df = pd.DataFrame(np.random.randn(100, 4), columns=['A', 'B', 'C', 'D'])


# In[137]:


df.head()


# In[138]:


# Create a DataFrame for categorical data with categories and their corresponding values
df2 = pd.DataFrame({'category': ['A', 'B', 'C'], 'values': [32, 43, 50]})


# In[139]:


df2


# In[140]:


import pandas as pd
import numpy as np
import plotly.graph_objs as go

# Sample DataFrame with random values
df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))

# Define different RGBA colors for each column
colors = [
    'rgba(255, 99, 132, 1)',    # red
    'rgba(54, 162, 235, 1)',    # blue
    'rgba(255, 206, 86, 1)',    # yellow
    'rgba(75, 192, 192, 1)'     # teal
]

# Create figure
fig = go.Figure()

# Add traces for each column with a different color
for i, col in enumerate(df.columns):
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[col],
        mode='lines+markers',
        name=col,
        line=dict(color=colors[i])
    ))

# Customize layout
fig.update_layout(
    title='Colored Graph',
    xaxis_title='x',
    yaxis_title='y',
    xaxis=dict(tickfont=dict(size=16, color='blue')),
    yaxis=dict(tickfont=dict(size=16, color='black')),
    title_font=dict(size=24, color='red')
)

# Show figure
fig.show()


# In[141]:


df = pd.DataFrame({
    'A': np.random.randn(50),
    'B': np.random.randn(50)
})

alpha = np.float64(1.0)
alpha_float = float(alpha)

color_str = f'rgba(255, 153, 51, {alpha_float})'
print(color_str)  # rgba(255, 153, 51, 1.0)

trace = go.Scatter(
    x=df['A'],
    y=df['B'],
    mode='markers',
    marker=dict(color=color_str)
)

layout = go.Layout(
    title='scatter plot',
    xaxis=dict(title='x', showgrid=True, zeroline=True),
    yaxis=dict(title='y', showgrid=True, zeroline=True),
    plot_bgcolor='grey',
    width=800,
    height=500
)

fig = go.Figure(data=[trace], layout=layout)
pyo.iplot(fig)



# In[142]:


import plotly.graph_objs as go

# Create an empty Figure object
fig = go.Figure()

# Loop over the specified columns to create a box plot for each
for col in ['A', 'B']:  # You can replace this list with df.columns.tolist() to include all columns
    fig.add_trace(go.Box(
        y=df[col],  # Data for the current column
        name=col,   # Name of the box plot (appears in the legend and x-axis)
        marker=dict(color='rgba(255, 153, 51, 1.0)')  # Set the marker color for the boxes
    ))

# Customize the layout of the figure
fig.update_layout(
    title='Box plots for A & B',  # Title of the plot
    xaxis=dict(title='Columns'),  # Label for the x-axis
    yaxis=dict(title='Values', zeroline=False),  # Label for the y-axis, no zero line
    plot_bgcolor='rgba(230,230,250, 0.5)',  # Background color with transparency
    width=800,  # Width of the plot in pixels
    height=500  # Height of the plot in pixels
)

# Display the figure
fig.show()



# In[143]:


df3 = pd.DataFrame ({'x': [1,2,3,4,5], 'y': [10,20,30,40,50], 'z':[500,400,300,200,200]})


# In[144]:


df3


# In[145]:


# Define the layout for the 3D surface plot using Plotly's Layout object
layout = go.Layout(
    title='3D service plot',  # Title of the entire plot
    scene=dict(               # Configure the 3D coordinate system (scene)
        xaxis=dict(title='X'),  # Label for the X-axis
        yaxis=dict(title='Y'),  # Label for the Y-axis
        zaxis=dict(title='Z'),  # Label for the Z-axis
    ),
    width=800,   # Set the width of the plot in pixels
    height=800   # Set the height of the plot in pixels
)

# Use cufflinks' iplot method to create a 3D surface plot from the DataFrame 'df'
df.iplot(kind='surface', layout=layout)
# - 'kind="surface"' specifies that the plot type is a 3D surface
# - 'layout=layout' applies the layout settings defined above, including titles and size


# In[130]:


import plotly.graph_objs as go
import pandas as pd

# Create a sample DataFrame with two columns A and B
data = {
    'A': [1, 2, 3, 4, 5],  # X-axis values
    'B': [5, 6, 7, 8, 9]   # Y-axis values
}
df = pd.DataFrame(data)

# Create a Scatter plot figure using Plotly
fig = go.Figure(
    go.Scatter(
        x=df['A'],         # Set x-axis data from column 'A'
        y=df['B'],         # Set y-axis data from column 'B'
        mode='markers'     # Display only markers (points)
    )
)

# Update the layout of the figure with titles and styling
fig.update_layout(
    title='Scatter Plot',        # Title of the plot
    xaxis_title='X values',      # Label for the x-axis
    yaxis_title='Y Values',      # Label for the y-axis
    template='plotly'            # Use Plotly's default styling template
)

# Show the interactive plot
fig.show()


# In[28]:


df.plot()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




