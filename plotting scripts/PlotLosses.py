from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns

def main():

    configs = ["init_content", "init_random"]

    for config in configs:
        log_dir = f"../logs/{config}"

        df = load_dataframe(log_dir)
        fig, axes = plt.subplots(1, 2)
        

        sns.lineplot(data=df.loc[:, "Content loss"], ax=axes[0], color='red')
        sns.lineplot(data=df.loc[:, "Style loss"], ax=axes[1])

        axes[0].set_title("Content loss")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")


        axes[1].set_title("Style loss")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Loss")

        # grid
        for ax in axes.flatten():
            ax.grid()

        plt.suptitle(f"{config}")
        plt.tight_layout()
        plt.savefig(f"../plots/losses_{config}.png")
    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")