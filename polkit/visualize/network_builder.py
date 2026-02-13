from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Literal

class NetworkBuilder:

    def __init__(self):

        self.G = nx.DiGraph()

    def build_network(self, layout_style:Literal["spring", "kamada", "circular"], matrix:np.ndarray, matrix_key:dict, digraph_data:pd.DataFrame):

        G = self._convert_matrix_to_digraph(matrix, matrix_key)
        pos = self._define_layout(G, layout_style)
        edges, arrows = self._build_edge_traces_with_annotations(G, pos)
        nodes = self._build_node_trace(G, pos, digraph_data)

        fig = go.Figure(
            data=[*edges, nodes],
            layout=go.Layout(
                title="Transition Network",
                showlegend=False,
                hovermode="closest",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                annotations=arrows,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=20)
            )
        )
        return fig

    def _convert_matrix_to_digraph(self, matrix:np.ndarray, matrix_key:dict):

        for state in matrix_key.keys():
            self.G.add_node(state)
        
        for from_state, i in matrix_key.items():
            for to_state, j in matrix_key.items():
                prob = matrix[i, j]
                if prob > .10 and to_state != from_state:
                    self.G.add_edge(from_state, to_state, weight=prob)
                
        return self.G
    
    def _define_layout(self, G:nx.DiGraph, layout_style:Literal["spring", "kamada", "circular"]):
        if layout_style == "spring":
            return nx.spring_layout(G)
        if layout_style == "kamada":
            return nx.kamada_kawai_layout(G)
        else:
            return nx.circular_layout(G)
        
    def _build_edge_traces_with_annotations(self, G:nx.DiGraph, pos:dict):
        edge_traces = []
        annotations = []

        for from_node, to_node, data in G.edges(data=True):
            x0, y0 = pos[from_node]
            x1, y1 = pos[to_node]
            weight = data["weight"]

            # Each edge gets a trace
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(
                        width=weight * 10,
                        color=f"rgba(150, 150, 150, {weight})"
                    ),
                    hoverinfo="none"
                )
            )

            annotations.append(dict(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=weight * 3,
                arrowcolor=f"rgba(150, 150, 150, {min(weight + 0.3, 1.0)})",
                opacity=min(weight + 0.2, 1)
            ))

        return edge_traces, annotations
    
    def _build_node_trace(self, G:nx.DiGraph, pos:dict, digraph_data:pd.DataFrame):
        label_colors = {
            "Anchor": "gold",
            "Habit": "steelblue",
            "Recurring": "mediumseagreen",
            "Transient": "salmon",
            "Unknown": "gray"
        }
        
        node_x, node_y, node_text, node_hover, node_colors = [], [], [], [], []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
        
            profile_row = digraph_data[digraph_data["Location ID"] == node]

            if not profile_row.empty:
                hover = profile_row["Hover"].iloc[0]
                classification = profile_row["Loyalty Label"].iloc[0]
                node_colors.append(label_colors[classification])
            else:
                hover = f"Location: {node}"
                node_colors.append("gray")

            node_hover.append(hover)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            hovertext=node_hover,
            hoverinfo="text",
            marker=dict(
                size=20,
                color=node_colors,
                line=dict(
                    width=2, color="white"
                )
            )
        )

        return node_trace