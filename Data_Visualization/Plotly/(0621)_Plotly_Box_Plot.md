# 📈 [0621] Plotly Box Plot Example

This notebook demonstrates how to create box plots for two DataFrame columns ('A' and 'B') using Plotly in Python.

```python
import plotly.graph_objs as go

fig = go.Figure()

for col in ['A', 'B']:  # Loop through specified columns to create box plots
    fig.add_trace(go.Box(
        y=df[col],
        name=col,
        marker=dict(color='rgba(255, 153, 51, 1.0)')  # Set marker color; can be customized per column
    ))

fig.update_layout(
    title='Box plots for A & B',
    xaxis=dict(title='Columns'),
    yaxis=dict(title='Values', zeroline=False),
    plot_bgcolor='rgba(230,230,250, 0.5)',
    width=800,
    height=500
)

fig.show()

