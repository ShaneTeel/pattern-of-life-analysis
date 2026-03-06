from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calplot

class ChartMaker:

    _LABEL_COLORS = {
        "Anchor": "gold",
        "Persistent": "steelblue",
        "Recurring": "mediumseagreen",
        "Transient": "salmon",
        "Unknown": "gray"
    }

    def create_gaps_gantt(self, gaps_df:pd.DataFrame):
        gap_fig = px.timeline(gaps_df, x_start="start", x_end="end", y="Gap ID", color="duration_hours", title="All Gaps (>24 hours)")
        gap_fig.update_layout(xaxis=dict(tickformat="%Y-%m-%d %H:%M"))
        return gap_fig
    
    def create_calendar_heatmap(self, pfs:pd.DataFrame, user_id:str):
        dt_series = pfs["datetime"].dt.normalize().dt.tz_localize(None).value_counts().sort_index()
        calplot_fig, _ = calplot.calplot(
            data=dt_series, 
            suptitle=f"User {user_id}'s Activity by Date",
            yearlabel_kws={"fontname": "sans-serif"}
        )
        return calplot_fig
    
    def create_time_wheel(self, pfs:pd.DataFrame, user_id:str):
        pfs["hour"] = pfs["datetime"].dt.hour * 15
        grouped = pfs.groupby("hour").size().reset_index(name="Frequency")
        polar_fig = px.bar_polar(
            data_frame=grouped,
            r="Frequency",
            theta="hour",
            title=f"User {user_id}'s Activity by Hour",
            color="Frequency",
            template="plotly_dark",
            color_continuous_scale="Viridis"
        )
        polar_fig.update_layout(
            polar=dict(
                angularaxis=dict(
                    tickmode='array',
                    tickvals=[i * 15 for i in range(24)],
                    ticktext=[f"{h}:00" for h in range(24)],
                    direction="clockwise"
                )
            )
        )
        return polar_fig

    def create_day_of_week_chart(self, df:pd.DataFrame, user_id:str):
        df["day_of_week"] = df["datetime"].dt.day_name()
        day_of_week_info = df.groupby("day_of_week").size().reset_index(name="Frequency")
        cat_order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

        bar_fig = px.bar(
            data_frame=day_of_week_info,
            x=day_of_week_info["day_of_week"],
            y=day_of_week_info["Frequency"],
            category_orders={"day_of_week": cat_order},
            title=f"User {user_id}'s Activity by Day of Week"

        )
        return bar_fig
    
    def create_location_profile_chart(self, data:pd.DataFrame):
        col_names = [
            ["Visit Count", "Recency", "Depth", "Visit Count"], 
            ["Arrival Certainty", "Dwell Certainty", "Gap Certainty", "Arrival Certainty"],
            ["Spatial Focus"]
        ]

        fig = make_subplots(
            rows=2,
            cols=2,
            row_heights=[0.7, 0.3],
            specs=[
                [{"type": "polar"}, {"type": "polar"}],
                [{"type": "indicator", "colspan": 2}, None]
            ]
        )

        fig.add_trace(
            go.Scatterpolar(
                r=list(data.loc[0, col_names[0]].values) + [data.loc[0, col_names[0]].values[0]],
                theta=col_names[0],
                fill="toself",
                name="Maturity",
                mode="lines"
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatterpolar(
                r=list(data.loc[0, col_names[1]].values) + [data.loc[0, col_names[1]].values[0]],
                theta=col_names[1],
                fill="toself",
                name="Predictability",
                mode="lines"
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Indicator(
                value=data.loc[0, col_names[2][0]],
                mode="gauge",
                title={"text": "Spatial Focus"},
                gauge={"axis": {"range": [0, 1]}}
            ),
            row=2, col=1
        )

        frames = []
        steps = []

        for _, loc in data.iterrows():
            loc_id = str(loc["Location ID"])
            frames.append(
                go.Frame(
                    data=[
                        go.Scatterpolar(
                            r=list(loc[col_names[0]].values) + [loc[col_names[0]].values[0]],
                            theta=col_names[0],
                            fill="toself",
                            mode="lines"
                        ),
                        go.Scatterpolar(
                            r=list(loc[col_names[1]].values) + [loc[col_names[1]].values[0]],
                            theta=col_names[1],
                            fill="toself",
                            mode="lines"
                        ),
                        go.Indicator(
                            value=loc[col_names[2][0]],
                            mode="gauge",
                            gauge={"axis": {"range": [0, 1]}}
                        )
                    ],
                    name=loc_id,
                    layout={
                        "title": 
                        {
                            "text": loc["Hover"],
                            "x": 0.46,
                            "xanchor": "center"
                        }
                    }
                )
            )
            steps.append(
                {
                    "method": "animate",
                    "args": [
                        [loc_id], 
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }
                    ],
                    "label": loc["Location ID"]
                }
            )

        fig.frames = frames

        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        sliders = [
            {
                "active": 0,
                "currentvalue": {"prefix": "Location: "},
                "pad": {"t": 50},
                "steps": steps
            }
        ]

        fig.update_layout(
            sliders=sliders,
            polar=
            {
                "angularaxis": {"rotation": 90},
                "radialaxis": {"range": [0, 1], "tickvals": [0, 0.2, 0.4, 0.6, 0.8, 1.0]}
            },
            polar2=
            {
                "angularaxis": {"rotation": 90},
                "radialaxis": {"range": [0, 1], "tickvals": [0, 0.2, 0.4, 0.6, 0.8, 1.0]}
            },
            template="plotly_dark"
        )

        return fig
    
    def create_stability_gantt(self, df:pd.DataFrame):

        fig = go.Figure()

        for _, loc in df.iterrows():
            fig.add_trace(go.Scatter(
                x=[loc["First Seen"], loc["Last Seen"], None],
                y=[loc["Location ID"], loc["Location ID"], None],
                mode="lines+markers",
                name="Temporal Stability",
                hoverinfo="all",
                marker=dict(color=self._LABEL_COLORS[loc["Maturity Label"]], symbol="diamond"),
                line=dict(width=3)
            ))

        fig.update_layout(
            xaxis=dict(tickformat="%Y-%m-%d %H:%M"), 
            title="Location Temporal Stability",
            xaxis_title="Datetimes",
            yaxis_title="Location IDs",
            showlegend=False,
        )

        return fig