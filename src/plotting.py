import matplotlib.pyplot as plt
import matplotlib.patches as mp
import torch


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

        ax.plot(*x_hat.T, "-o", color=f"C{i}")
        ax.plot(*loc_q[i].T, linestyle="--", color=f"C{i}")

        for j in range(100, mini_batch_length[i], 50):
            ax.plot(*loc_q[i, j, :], "x", color=f"C{i}")
            ax.add_patch(
                plt.Circle(loc_q[i, j, :], 2 * scale_q[i], fill=False, color=f"C{i}")
            )



def plot_wifi(model, ax=None):

    if ax is None:
        ax = plt.gca()

    with torch.no_grad():
        wifi_locations = model.wifi_location_q.numpy()
        scale = model.wifi_location_log_sigma_q.exp().numpy()

    for i in range(scale.shape[0]):

        ax.plot(*wifi_locations[i, :], ".", color=f"C{i}")
        ax.add_patch(
            mp.Ellipse(wifi_locations[i, :], *(2*scale[i]), fill=False, color=f"C{i}", alpha=0.2)
        )

def plot_beacons(model, ax=None):

    if ax is None:
        ax = plt.gca()

    with torch.no_grad():
        beacon_locations = model.beacon_location_q.numpy()
        scale = model.beacon_location_log_sigma_q.exp().numpy()

    for i in range(scale.shape[0]):

        ax.plot(*beacon_locations[i, :], ".", color=f"C{i}")
        ax.add_patch(
            mp.Ellipse(beacon_locations[i, :], *(2*scale[i]), fill=False, color=f"C{i}")
        )