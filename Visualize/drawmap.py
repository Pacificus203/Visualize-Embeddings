import pandas as pd
from embedding6 import embed
import numpy as np
import umap
import plotly.express as px
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
import textwrap




import random



def random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


def drawmap():
    LABEL = ['blueprints','workload','software - microservices, sdks, libraries, frameworks', 'accelerated computing, infra software, networking', 'key isvs and partners']
    
    a = input("0,1,2,3,4: ")
    LABEL = LABEL[int(a)]     
    X,Y = embed()  # shape (170, 4090)
    
    
    X = np.array(X)
     
    Y = Y.iloc[:-1]
    unique_keys = Y[LABEL].unique()

    color_map = {k: random_color() for k in unique_keys}

    Y = pd.concat(
        [Y[LABEL], Y[LABEL].map(color_map).rename("color")],
        axis=1
    )    


        
    Y[LABEL] = (
    Y[LABEL]
    .astype(str)
    .str.rstrip()
    )
        

    # Khởi tạo UMAP
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        metric="cosine",   # rất hợp cho embedding
        random_state=42
    )

    # Giảm chiều
    X_2d = reducer.fit_transform(X)
    
    # Tạo DataFrame cho plotly

    df_plot = pd.DataFrame(X_2d, columns=["x", "y"])
    
    user = df_plot.iloc[-1] 
    df_plot = df_plot.iloc[:-1]

    
    df_plot = pd.concat(
        [df_plot,Y],
        axis=1
    )
    
    Y = (
    Y[[LABEL, "color"]]
    .drop_duplicates(subset=LABEL, keep="first")
    .sort_values(LABEL)
    .reset_index(drop=True)
    )

    # Vẽ plotly
    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        title="UMAP projection",
    )

    
    fig = go.Figure()

# Vẽ tất cả điểm
    fig.add_trace(
        go.Scatter(
            x=df_plot["x"],
            y=df_plot["y"],
            mode="markers",
            marker=dict(
                size=6,
                color=df_plot["color"]   # 👈 dùng cột color
            ),
            hovertext=df_plot[LABEL],      # 👈 tên label
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "x: %{x:.3f}<br>"
                "y: %{y:.3f}"
                "<extra></extra>"        # 👈 bỏ dòng 'trace 0'
            )
        )
    )
    # ve user
    fig.add_trace(
        go.Scatter(
            x=[user.iloc[0]],
            y=[user.iloc[1]],
            mode="markers",
            marker=dict(
                size=15,
                opacity=0.8,
                color='black'   # 👈 dùng cột color
            ),
            hovertemplate=(
                "<b>USER</b><br>"
                "x: %{x:.3f}<br>"
                "y: %{y:.3f}"
                "<extra></extra>"        # 👈 bỏ dòng 'trace 0'
            )
        )
    )

    
    
    annotations = []
    y_start = 1.0
    y_step = 0.02   # khoảng cách giữa các dòng

    for i, row in Y.iterrows():
        annotations.append(
            dict(
                x=1.1,                # 👈 ngoài mép phải
                y=y_start - i * y_step,
                xref="paper",
                yref="paper",
                text=f"<span style='color:{row['color']}'>●</span> {row[LABEL]}",
                showarrow=False,
                align="right",
                font=dict(size=10),
                xanchor="left"
            )
        )
    
    
    for label,g in df_plot.groupby(LABEL):
        if len(label) < 2:
            continue

        coords = g[["x", "y"]].values

        db = DBSCAN(
            eps=0.3,
            min_samples=2
        ).fit(coords)

        labels = db.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        centroids = []
        for c in set(labels):
            if c == -1:
                continue  # bỏ noise

            pts = coords[labels == c]
            center = pts.mean(axis=0)
            
            # 🔥 khoảng cách tới centroid
            dists = np.linalg.norm(pts - center, axis=1)
            max_dist = dists.max()

            centroids.append({
                LABEL: label,
                "cluster": c,
                "x_center": center[0],
                "y_center": center[1],
                "size": len(pts),
                'max_dist': max_dist,
                "color": g["color"].iloc[0]
            })

        if(n_clusters!=0):
            centroids_df = pd.DataFrame(centroids, columns=[LABEL,'cluster', 'x_center', 'y_center', 'size','color','max_dist'])
            centroids_df["max_dist"] = centroids_df["max_dist"] * 100 + 20
            
            
            fig.add_trace(
                go.Scatter(
                    x=centroids_df["x_center"],
                    y=centroids_df["y_center"],
                    mode="markers",
                    marker=dict(
                        size=centroids_df["max_dist"],
                        sizemode="diameter",
                        color=centroids_df["color"],
                        opacity=0.5,
                        line=dict(width=1, color="black")
                    ),
                    text=centroids_df[LABEL],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Cluster size: %{customdata}"
                        "<extra></extra>"
                    ),
                    customdata=centroids_df["size"],
                    name="Centroids"
                )
            )
        
        

    # fig.update_traces(marker=dict(size=8))
    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        width=1800,
        height=3000,
        showlegend=False,
        annotations=annotations,
        margin=dict(r=900, b=2000),
        title=LABEL
    )
    fig.update_yaxes(scaleanchor=None)
    fig.show()

    
drawmap()