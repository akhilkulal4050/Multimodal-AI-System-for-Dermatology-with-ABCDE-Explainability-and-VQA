import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "CMC_utils"))

import pandas as pd
from CMC_utils.plots import load_experiments_results

from dash import Dash, dcc, html, Input, Output
import plotly.express as px


figure_height = 600

# results_path = "/Volumes/Aaron SSD/UCBM/Projects/multiNAIM_VIPVIZA/multiNAIM_classification"
# results_path = "/Volumes/Aaron SSD/UCBM/Projects/multiNAIM_VIPVIZA/multiNAIM_lrxm"
# results_path = "/Users/camillocaruso/LocalDocuments/outputs_pembro_classification/pembro_5y_42_classification_with_missing_generation/stratifiedkfold_holdout"
# results_path = "/Users/camillocaruso/LocalDocuments/code_outputs/VIPVIZA_lrxm/VIPVIZA_score2_42_multimodal_joint_fusion_regression_with_missing_generation/stratifiedkfold_holdout"
# results_path = "/Volumes/Aaron SSD/UCBM/Projects/multiNAIM_ADNI/ADNI_diagnosis"
# results_path = "/Users/camillocaruso/LocalDocuments/code_outputs/ADNI_prognosis"
results_path = "/Users/camillocaruso/LocalDocuments/code_outputs/AI4Covid"
# results_path = "/Volumes/Aaron SSD/UCBM/Projects/multiNAIM_VIPVIZA/multiNAIM_score2"
# results_path = "/Users/camillocaruso/LocalDocuments/multiNAIM_regression1"
task = "classification"
separate_experiments = False
multimodal = True

all_results = load_experiments_results(results_path, separate_experiments=separate_experiments, task=task, multimodal=multimodal)

mean_cols = [col for col in all_results.columns if col.endswith("_mean")]
metrics_cols = [col.replace("_mean", "") for col in mean_cols]

mean_results = all_results[mean_cols + ["class"]].rename(columns={col: col.replace("_mean", "") for col in mean_cols})

std_cols = [col for col in all_results.columns if col.endswith("_std")]
std_results = all_results[std_cols + ["class"]].rename(columns={col: col.replace("_std", "") for col in std_cols})

del mean_cols, std_cols, all_results, results_path

mean_results = mean_results.reset_index()
std_results = std_results.reset_index()

other_cols = [col_name for col_name in mean_results.columns if col_name not in metrics_cols + ["train_percentage", "test_percentage"]]
mean_results.loc[:, other_cols] = mean_results[other_cols].astype(str)

info_cols = {col_name: sorted(mean_results[col_name].unique().tolist()) for col_name in mean_results.columns if col_name not in metrics_cols + ["test_percentage", "experiment"]}
info_cols["metrics"] = metrics_cols

df1 = mean_results.copy()
dbs_mask1 = df1["db"] == info_cols["db"][0]
classes_mask1 = df1["class"] == info_cols["class"][-1]
missing_mask1 = df1["missing_strategy"] == info_cols["missing_strategy"][0]
fusion_mask1 = df1["fusion_strategy"] == info_cols["fusion_strategy"][0]
train_percentages_mask1 = df1["train_percentage"] == info_cols["train_percentage"][0]

mask1 = pd.concat([ missing_mask1, dbs_mask1, classes_mask1, fusion_mask1, train_percentages_mask1], axis=1).all(axis=1)  # fusion_mask1
df1 = df1.loc[mask1]

app = Dash(__name__)

fig = px.line(x=df1.loc[:, "test_percentage"], y=df1.loc[:, info_cols["metrics"][0]], color=df1.loc[:, 'experiment'], height=figure_height)

app.layout = html.Div([
    html.H1('Experiments'),
    html.Div([
        html.Div(children=[
            html.H2('Models'),
            dcc.Checklist(
                id="models_checklist",
                options=info_cols["model"],
                value=info_cols["model"],
                inline=True,
                style={'display': 'flex', 'flex-direction': 'column'}
            ),
        ], style={'padding': 10, 'flex': 1}),
        html.Div(children=[
            html.H2('DBs'),
            dcc.RadioItems(
                id="dbs_radio",
                options=info_cols["db"],
                value=info_cols["db"][0],
                inline=True,
                style={'display': 'flex', 'flex-direction': 'column'}
            ),
            html.H2('Classes'),
            dcc.RadioItems(
                id="classes_radio",
                options=info_cols["class"],
                value=info_cols["class"][-1],
                inline=True,
                style={'display': 'flex', 'flex-direction': 'column'}
            ),
            html.H2('Missing strategies'),
            dcc.RadioItems(
                id="missing_strategy_radio",
                options=info_cols["missing_strategy"],
                value=info_cols["missing_strategy"][0],
                inline=True,
                style={'display': 'flex', 'flex-direction': 'column'}
            ),
        ], style={'padding': 10, 'flex': 1}),
        html.Div(children=[
            html.H2('Locked'),
            dcc.Dropdown(
                id="experiments_dropdown",
                options=sorted(mean_results["experiment"].unique().tolist()),
                value=[],
                multi=True,
                style={'display': 'flex', 'flex-direction': 'column'}),
            html.H2('Imputers'),
            dcc.Checklist(
                id="preprocessing_checklist",
                options=info_cols["imputer"],
                value=info_cols["imputer"],
                inline=True,
                style={'display': 'flex', 'flex-direction': 'column'}
            ),
            html.H2('Train percentages'),
            dcc.RadioItems(
                id="train_percentages_radio",
                options=info_cols["train_percentage"],
                value=info_cols["train_percentage"][0],
                inline=True,
                style={'display': 'flex', 'flex-direction': 'column'}
            ),
        ], style={'padding': 10, 'flex': 1}),
        html.Div(children=[
            html.H2('Fusion strategies'),
            dcc.Checklist(
                id="fusion_strategy_checklist",
                options=info_cols["fusion_strategy"],
                value=[info_cols["fusion_strategy"][0]],
                inline=True,
                style={'display': 'flex', 'flex-direction': 'column'}
             ),
            html.H2('Metrics'),
            dcc.RadioItems(
                id="metrics_radio",
                options=info_cols["metrics"],
                value=info_cols["metrics"][0],
                inline=True,
                style={'display': 'flex', 'flex-direction': 'column'}
            ),
            html.H2('Additional Options'),
            dcc.Checklist(
                id="std_check",
                options=["Visualize STD"],
                value=[],
                inline=True,
                style={'display': 'flex', 'flex-direction': 'column'}
            ),
        ], style={'padding': 10, 'flex': 1}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),

    html.Div(children=[
        dcc.Graph(id="graph", figure=fig),
    ], style={'display': 'flex', 'flex-direction': 'column'})
], style={'display': 'flex', 'flex-direction': 'column'})


@app.callback( Output("graph", "figure"),
               Input("models_checklist", "value"),
               Input("dbs_radio", "value"),
               Input("classes_radio", "value"),
               Input("missing_strategy_radio", "value"),
               Input("preprocessing_checklist", "value"),
               Input("train_percentages_radio", "value"),
               Input("fusion_strategy_checklist", "value"),
               Input("experiments_dropdown", "value"),
               Input("metrics_radio", "value"),
               Input("std_check", "value"))
def update_line_chart(models, db, classes, missings, preprocessings, train_percentages, fusions, experiments, metric, std_vis):
    df = mean_results.copy()

    if not models:
        models = [info_cols["models"][0]]
    if not preprocessings:
        preprocessings = [info_cols["imputer"][0]]
    if not train_percentages:
        train_percentages = info_cols["train_percentage"][0]
    if not fusions:
        fusions = [info_cols["fusion_strategy"][0]]

    if not classes:
        classes = info_cols["class"][0]
    if not db:
        db = info_cols["db"][0]

    model_mask = df["model"].isin(models)
    dbs_mask = df["db"] == db
    classes_mask = df["class"] == classes
    missings_mask = df["missing_strategy"] == missings
    preprocessings_mask = df["imputer"].isin(preprocessings)
    fusions_mask = df["fusion_strategy"].isin(fusions)
    train_percentages_mask = df["train_percentage"] == train_percentages

    mask = pd.concat([model_mask, dbs_mask, classes_mask, missings_mask, preprocessings_mask, fusions_mask, train_percentages_mask], axis=1).all(axis=1)

    experiments_mask = pd.concat([ df["experiment"].isin(experiments), dbs_mask, classes_mask, missings_mask, train_percentages_mask], axis=1).all(axis=1)
    mask = pd.concat([mask, experiments_mask], axis=1).any(axis=1)

    df = df.loc[mask]
    if std_vis:
        df2 = std_results.copy().loc[mask]
        figr = px.line(x=df.loc[:, "test_percentage"], y=df.loc[:, metric], error_y=df2.loc[:, metric], color=df.loc[:, 'experiment'], height=figure_height)
    else:
        figr = px.line(x=df.loc[:, "test_percentage"], y=df.loc[:, metric], color=df.loc[:, 'experiment'], height=figure_height)

    return figr


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
