import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter


# function to plot line graphs
def plot_metric(data, metric, title, y_label, log_scale=False):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='maze', y=metric, hue='algorithm', marker='o')
    plt.title(title)
    plt.xlabel("Maze Size")
    plt.ylabel(y_label)
    plt.xticks(rotation=45)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
    ax.ticklabel_format(style='plain', axis='y')
    if log_scale:
        ax.set_yscale('log')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv("results/results_combined.csv")
    df['maze'] = df['maze'].str.split('-').str[0]
    df['cells'] = df['maze'].str.split('x').apply(lambda x: int(x[0]) * int(x[1]))

    # convert relevant columns to numeric
    numeric_columns = ['execution_time (s)', 'peak_memory (kB)', 'path_length', 'total_moves', 'epochs']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # aggregate results of the same algorithm and maze size
    df = df.groupby(['maze', 'algorithm', 'cells'], as_index=False)[numeric_columns].mean()
    df = df.sort_values(by=['cells', 'algorithm'])

    # configure seaborn
    sns.set_theme(style="whitegrid")

    plot_metric(df, "execution_time (s)", "Execution Time Comparison Across Mazes", "Time (s)", log_scale=True)
    plot_metric(df, "peak_memory (kB)", "Peak Memory Usage Comparison Across Mazes", "Memory (kB)")
    plot_metric(df, "path_length", "Path Length Comparison Across Mazes", "Path Length")

    df_search = df[df['total_moves'].notnull()]
    plot_metric(df_search, "total_moves", "Total Moves Comparison Across Mazes", "Total Moves", log_scale=True)

    df_mdp = df[df['epochs'].notnull()]
    plot_metric(df_mdp, "epochs", "Epochs Comparison Across Mazes", "Epochs")
