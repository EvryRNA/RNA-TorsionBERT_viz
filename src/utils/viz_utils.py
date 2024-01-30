from typing import Any

import pandas as pd


def clean_fig(fig):
    fig.update_annotations(font_size=10)
    params_axes = dict(
        showgrid=True,
        gridcolor="grey",
        linecolor="black",
        zeroline=False,
        linewidth=1,
        showline=True,
        mirror=True,
        gridwidth=1,
        griddash="dot",
        tickson="boundaries",
    )
    fig.update_yaxes(**params_axes)
    fig.update_xaxes(**params_axes)
    fig.update_layout(dict(plot_bgcolor="white"), margin=dict(l=10, r=5, b=10, t=20))
    param_marker = dict(opacity=1, line=dict(width=0.5, color="DarkSlateGrey"), size=6)
    fig.update_traces(marker=param_marker, selector=dict(mode="markers"))
    fig.update_layout(
        font=dict(
            family="Computer Modern",
            size=10,  # Set the font size here
        )
    )
    fig.update_yaxes(matches=None)
    return fig


def update_fig_box_plot(
    fig: Any,
) -> Any:
    fig.update_yaxes(matches=None, showticklabels=True)
    params_axes = dict(
        showgrid=True,
        gridcolor="#d6d6d6",
        linecolor="black",
        zeroline=False,
        linewidth=1,
        showline=True,
        mirror=True,
        gridwidth=1,
        griddash="dot",
        title=None,
    )
    fig.update_xaxes(**params_axes)
    fig.update_yaxes(**params_axes)
    fig.update_layout(dict(plot_bgcolor="white"), margin=dict(l=5, r=5, b=5, t=20))
    param_marker = dict(opacity=1, line=dict(width=0.5, color="DarkSlateGrey"), size=6)
    fig.update_traces(marker=param_marker, selector=dict(mode="markers"))
    fig.update_layout(
        font=dict(
            family="Computer Modern",
            size=12,  # Set the font size here
        )
    )
    fig.update_layout(
        legend=dict(
            orientation="v",
            bgcolor="#f3f3f3",
            bordercolor="Black",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(visible=True, showticklabels=True)
    return fig


def rename_angles(df: pd.DataFrame):
    return df.replace(
        {
            "alpha": r"$\alpha$",
            "beta": r"$\beta$",
            "gamma": r"$\gamma$",
            "delta": r"$\delta$",
            "epsilon": r"$\epsilon$",
            "zeta": r"$\zeta$",
            "chi": r"$\chi$",
            "eta": r"$\eta$",
            "theta": r"$\theta$",
        },
    )
