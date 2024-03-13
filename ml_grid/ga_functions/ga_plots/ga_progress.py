import matplotlib.pyplot as plt


def plot_generation_progress_fitness(
    generation_progress_list, pop_val, g_val, nb_val, file_path
):
    fig = plt.figure()
    ax = plt.axes()

    x = [x for x in range(0, len(generation_progress_list))]
    ax.plot(x, generation_progress_list)
    plt.savefig(
        file_path
        + "/logs/"
        + "figures/"
        + "best_pop="
        + str(pop_val)
        + "_g="
        + str(g_val)
        + "_nb="
        + str(nb_val)
        + ".png",
        bbox_inches="tight",
    )
