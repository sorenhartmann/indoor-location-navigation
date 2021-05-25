import matplotlib.pyplot as plt
import matplotlib.patches as mp
from scipy.interpolate import interp1d
from seaborn.palettes import color_palette
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

from tqdm import tqdm
import functools
import pandas as pd
from src.data.datasets import (
    SiteDataset,
    FloorDataset,
    TraceData,
    raw_path,
    RAW_FILE_NAME,
)
import seaborn as sns

from src.models.initial_model import InitialModel

def plot_observed_trace(trace_data, ax=None):

    if ax is None:
        ax = plt.gca()

    time, position, _, _ = trace_data[0]

    position_is_observed = (~position.isnan()).any(-1).nonzero().flatten()
    start, end = time[position_is_observed[[0, -1]]]

    tt = torch.linspace(start, end, 300)
    xx = interp1d(time[position_is_observed], position[position_is_observed, 0])(tt)
    yy = interp1d(time[position_is_observed], position[position_is_observed, 1])(tt)

    points = torch.tensor([xx, yy]).T.reshape(-1, 1, 2)
    segments = torch.cat([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(tt.min(), tt.max())
    lc = LineCollection(segments, cmap="viridis", norm=norm)
    lc.set_array(tt)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    ax.scatter(*position[position_is_observed].T)

    return line


@functools.lru_cache(20)
def _get_wifi_strengths(floor_data, bssid):

    wifi_strengths = {
        "x": [],
        "y": [],
        "rssi": [],
        "bssid": [],
    }

    for trace in floor_data.traces:

        wifi_data = trace.data["TYPE_WIFI"][trace.data["TYPE_WIFI"]["bssid"] == bssid]
        interpolated_wifi = (
            pd.DataFrame(
                {
                    "x": trace.data["TYPE_WAYPOINT"]["x"],
                    "y": trace.data["TYPE_WAYPOINT"]["y"],
                    "rssi": wifi_data["rssi"].groupby("time").mean(),
                }
            )
            .interpolate("time")
            .bfill()
            .reindex(trace.data["TYPE_WAYPOINT"].index)
        )

        wifi_strengths["x"].extend(interpolated_wifi["x"])
        wifi_strengths["y"].extend(interpolated_wifi["y"])
        wifi_strengths["rssi"].extend(interpolated_wifi["rssi"])
        wifi_strengths["bssid"].extend(len(interpolated_wifi) * [bssid])

    return pd.DataFrame(wifi_strengths)


def plot_observed_wifi(floor_data, wifi_index, ax=None, color = "grey", hue = "rssi", with_scatter = True):

    if ax is None:
        ax = plt.gca()

    if not hasattr(floor_data, "bssids_"):
        floor_data._generate_tensors()

    bssid = floor_data.bssids_[wifi_index]
    wifi_strengths_df = _get_wifi_strengths(floor_data, bssid)

    if with_scatter:
        points = sns.scatterplot(
            data=wifi_strengths_df,
            x="x",
            y="y",
            size="rssi",
            hue="rssi" if hue == "rssi" else None,
            ax=ax,
            legend=False,
            color = color
        )
        

    sns.kdeplot(
        data=wifi_strengths_df,
        x="x",
        y="y",
        weights="rssi",
        color=color,
        alpha=0.5,
        ax=ax,
    )

    if with_scatter:
        return points


def _get_beacon_strengths(floor_data, uuids):

    beacon_strengths = {
        "x": [],
        "y": [],
        "uuid": [],
        "distance": [],
    }

    for trace in floor_data.traces:

        if "TYPE_BEACON" not in trace.data:
            continue
        
        for uuid, beacon_data in trace.data["TYPE_BEACON"].groupby("uuid"):

            if uuid not in uuids:
                continue

            interpolated_beacon = (
                pd.DataFrame(
                    {
                        "x": trace.data["TYPE_WAYPOINT"]["x"],
                        "y": trace.data["TYPE_WAYPOINT"]["y"],
                        "distance": beacon_data["distance"].groupby("time").mean(),
                    }
                )
                .interpolate("time")
                .bfill()
                .reindex(trace.data["TYPE_WAYPOINT"].index)
            )

            beacon_strengths["x"].extend(interpolated_beacon["x"])
            beacon_strengths["y"].extend(interpolated_beacon["y"])
            beacon_strengths["distance"].extend(interpolated_beacon["distance"])
            beacon_strengths["uuid"].extend(len(interpolated_beacon) * [uuid])

    return pd.DataFrame(beacon_strengths)


def plot_observed_beacon(floor_data, beacon_index, ax=None, **kwargs):

    if not hasattr(floor_data, "beacon_ids_"):
        floor_data._generate_tensors()

    if not hasattr(beacon_index, "__iter__"):
        beacon_index = [beacon_index]

    uuids = [floor_data.beacon_ids_[i] for i in beacon_index]
    beacon_strengths_df = _get_beacon_strengths(floor_data, uuids)
    norm = LogNorm(vmin = beacon_strengths_df["distance"].min(), vmax = beacon_strengths_df["distance"].max())

    aspect = floor_data.info["map_info"]["width"] / floor_data.info["map_info"]["height"]

    return sns.relplot(
        data=beacon_strengths_df,
        x="x",
        y="y",
        kind="scatter",
        size="distance",
        hue="distance",
        col="uuid",
        hue_norm=norm,
        size_norm=norm,
        aspect=aspect,
        **kwargs
    )


def plot_traces(model, mini_batch, ax=None):

    if ax is None:
        ax = plt.gca()

    mini_batch_index = mini_batch[0]
    mini_batch_length = mini_batch[1]
    mini_batch_position = mini_batch[3]
    mini_batch_position_mask = mini_batch[4]

    with torch.no_grad():
        loc_q, scale_q = model.guide(*mini_batch)
        loc_q[loc_q == 0] = float("nan")

    for i in range(len(mini_batch_index)):
        x_hat = mini_batch_position[i, mini_batch_position_mask[i], :]

        ax.plot(*x_hat.T, "-o", color=f"C{i}", label=f"trace {mini_batch_index[i]}")
        ax.plot(*loc_q[i].T, linestyle="--", color=f"C{i}")

        for j in mini_batch_position_mask[i].nonzero().flatten():
            ax.plot(*loc_q[i, j, :], "x", color=f"C{i}")
            ax.add_patch(
                plt.Circle(loc_q[i, j, :], 2 * scale_q[i], fill=False, color=f"C{i}")
            )

import numpy as np
def plot_diff_x_and_xhat(model, mini_batch, ax=None):

    if ax is None:
        ax = plt.gca()

    mini_batch_index = mini_batch[0]
    mini_batch_length = mini_batch[1]
    mini_batch_position = mini_batch[3]
    mini_batch_position_mask = mini_batch[4]

    with torch.no_grad():
        loc_q, scale_q = model.guide(*mini_batch)
        loc_q[loc_q == 0] = float("nan")

    x_hat_imputed = np.zeros((max(mini_batch_length),2))
    for i in range(len(mini_batch_index)):
        x_hat = mini_batch_position[i, mini_batch_position_mask[i], :]
        time = np.linspace(0, (max(mini_batch_length)-1)/10,max(mini_batch_length))
        x_hat_imputed[:,0] = np.interp(time, time[mini_batch_position_mask[i]], x_hat[:,0].numpy())
        x_hat_imputed[:,1] = np.interp(time, time[mini_batch_position_mask[i]], x_hat[:,1].numpy())
        x = loc_q[i].numpy()
        #x = loc_q[i, mini_batch_position_mask[i], : ]
        dist = np.sqrt(((x_hat_imputed-x)**2).sum(1))

        ax.plot(time, dist, "-", linewidth=3,color=f"C{i}", label=f"trace {mini_batch_index[i]}")
        for j in mini_batch_position_mask[i].nonzero().flatten():
            ax.plot(time[j],dist[j],markersize = 8, marker = "o", color=f"C{i}")



def get_wifi_ids(model, scale_threshold = 20):
    with torch.no_grad():
        wifi_locations = model.wifi_location_q.numpy()
        scale = model.wifi_location_log_sigma_q.exp().numpy()
    
    wifi_ids = []

    for i in range(scale.shape[0]):
        if(any(scale[i] > scale_threshold)):
            continue
        wifi_ids.append(i)
    
    return pd.DataFrame({
        "wifi_id":wifi_ids,
        "scalex": scale[wifi_ids,0],
        "scaley": scale[wifi_ids,1],
        "scale_combined": scale[wifi_ids,:].sum(1),
    }).sort_values("scale_combined")
    

def plot_wifi(model, ax=None, scale_thresshold = 1000, alpha = 0.2, wifi_ids = None, color=None):

    if ax is None:
        ax = plt.gca()
    
    wifi_ids_tmp = []    
    
    with torch.no_grad():
        wifi_locations = model.wifi_location_q.numpy()
        scale = model.wifi_location_log_sigma_q.exp().numpy()

    for i in range(scale.shape[0]):
        if(wifi_ids is not None and i not in wifi_ids):
            continue
        if(wifi_ids is None and any(scale[i] > scale_thresshold)):
            continue
        ax.plot(*wifi_locations[i, :], ".", color=f"C{i}")
        if color is None:
            color_=f"C{i}"
        else:
            color_=color
        ax.add_patch(
            mp.Ellipse(
                wifi_locations[i, :],
                *(2 * scale[i]),
                fill=False,
                color=color_,
                alpha=0.8,
            )
        )
        wifi_ids_tmp.append(i)
    return(wifi_ids_tmp)

def plot_emperical_and_infered_wifi(model,floor_data, wifi_ids, ax = None, with_scatter = True):

    if ax is None:
        ax = plt.gca()

    with torch.no_grad():
        wifi_locations = model.wifi_location_q.numpy()
        scale = model.wifi_location_log_sigma_q.exp().numpy()

    for i in wifi_ids:
        ax.plot(*wifi_locations[i, :], "*", markersize = 10,color=f"C{i}")
        ax.add_patch(
            mp.Ellipse(
                wifi_locations[i, :],
                *(2 * scale[i]),
                fill=False,
                color=f"C{i}",
                alpha=0.8,
            )
        )
        plot_observed_wifi(floor_data, i, ax=ax, color = f"C{i}", hue = f"C{i}", with_scatter=with_scatter)

def plot_beacons(model, ax=None):

    if ax is None:
        ax = plt.gca()

    with torch.no_grad():
        beacon_locations = model.beacon_location_q.numpy()
        scale = model.beacon_location_log_sigma_q.exp().numpy()

    for i in range(scale.shape[0]):

        ax.plot(*beacon_locations[i, :], ".", color=f"C{i}")
        ax.add_patch(
            mp.Ellipse(
                beacon_locations[i, :], *(2 * scale[i]), fill=False, color=f"C{i}"
            )
        )


if __name__ == "__main__":

    site_id = "5a0546857ecc773753327266"
    floor_id = "B1"
    floor_data = FloorDataset(site_id, floor_id, wifi_threshold=200, sampling_interval=100, include_wifi=False, include_beacon=False)
    initial_model = InitialModel(floor_data)

    mini_batch = floor_data[torch.tensor([2,5,10,20, 30])]
    mini_batch_index = mini_batch[0]
    mini_batch_length = mini_batch[1]
    mini_batch_time = mini_batch[2]
    mini_batch_position = mini_batch[3]
    mini_batch_position_mask = mini_batch[4]


    plot_diff_x_and_xhat(initial_model, mini_batch, ax=None)
