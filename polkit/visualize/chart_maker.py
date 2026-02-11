from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calplot

class ChartMaker:

    def __init__(self):

        pass

    def create_gaps_gantt(self, gaps_df:pd.DataFrame):
        gap_fig = px.timeline(gaps_df, x_start="start", x_end="end", y="Gap ID", color="duration_hours", title="All Gaps (>24 hours)")
        gap_fig.update_layout(xaxis=dict(tickformat="%Y-%m-%d %H:%M"))
        return gap_fig
    
    def create_calendar_heatmap(self, pfs:pd.DataFrame, user_id:str):
        dt_series = pfs["datetime"].dt.normalize().dt.tz_localize(None).value_counts().sort_index()
        calplot_fig, _ = calplot.calplot(
            data=dt_series, 
            suptitle=f"User {user_id}'s Activty by Date",
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

        diamond_fig = px.line_polar(
            data_frame=data,
            r="Score",
            theta="Metric",
            title="Profile Charts",
            animation_frame="Location ID",
            template="plotly_dark",
            line_close=True,
            range_r=[0, 1]
        )

        for i, frame in enumerate(diamond_fig.frames):
            frame.layout.title.text = data.loc[i, "Hover"]

        diamond_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        diamond_fig.update_traces(fill="toself")

        return diamond_fig
    
    def create_stability_gantt(self, df:pd.DataFrame):
        label_colors = {
            "Anchor": "gold",
            "Habit": "steelblue",
            "Recurring": "mediumseagreen",
            "Transient": "salmon",
            "Unknown": "gray"
        }

        fig = go.Figure()

        for _, loc in df.iterrows():
            fig.add_trace(go.Scatter(
                x=[loc["First Seen"], loc["Last Seen"], None],
                y=[loc["Location ID"], loc["Location ID"], None],
                mode="lines+markers",
                name="Temporal Stability",
                hoverinfo="all",
                marker=dict(color=label_colors[loc["Label"]], symbol="diamond"),
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