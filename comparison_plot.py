import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to plot comparison between Benchmark and Experimental models
def plot_comparison(df_metrics):
    plt.style.use('ggplot')
    n_metrics = len(df_metrics['Metric'])
    ind = np.arange(n_metrics)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(ind - width/2, df_metrics['Benchmark'], width, label='Benchmark', color='skyblue')
    bars2 = ax.bar(ind + width/2, df_metrics['Experimental'], width, label='Experimental', color='salmon')

    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Evaluation Metrics')
    ax.set_xticks(ind)
    ax.set_xticklabels(df_metrics['Metric'])
    ax.legend()

    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()
    plt.savefig('./imgs/comparison_metrics.png')
    plt.show()

# Function to calculate and plot percentage difference between models
def plot_percentage_difference(df_metrics):
    fig, ax = plt.subplots(figsize=(10, 6))
    ind = np.arange(len(df_metrics['Metric']))
    width = 0.35
    bars = ax.bar(ind, df_metrics['Percentage Improvement'], width, color='lightgreen')

    ax.set_ylabel('Percentage Improvement (%)')
    ax.set_title('Percentage Difference Between Experimental and Benchmark Models')
    ax.set_xticks(ind)
    ax.set_xticklabels(df_metrics['Metric'])

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    ax.axhline(0, color='grey', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('./imgs/percentage_difference.png')
    plt.show()

# Function to calculate percentage improvement and generate visualizations
def analyze_results():
    from results_data import data  # Import data from the second file
    df_metrics = pd.DataFrame(data)

    # Calculate the difference and percentage improvement
    df_metrics['Difference'] = df_metrics['Experimental'] - df_metrics['Benchmark']
    df_metrics['Percentage Improvement'] = (df_metrics['Difference'] / df_metrics['Benchmark']) * 100

    # Plot comparison and percentage difference
    plot_comparison(df_metrics)
    plot_percentage_difference(df_metrics)

if __name__ == "__main__":
    analyze_results()
