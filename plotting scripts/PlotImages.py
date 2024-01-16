from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns

def main():
    log_dir = "../logs/init_random"

    df = load_dataframe(log_dir)
    
    num_imgs = df.shape[0]

    for i in range(num_imgs):
        img = df.loc[i, "Image"]

        plt.imshow(img)
        plt.tight_layout()
        plt.axis("off")
        plt.savefig(f"../plots/init_random/{i}.png")

    return  
    print(df)

    fig, axes = plt.subplots(1, 2)
    

    sns.lineplot(data=df.loc[:, ["train accuracy", "test accuracy"]], ax=axes[0], markers=True)
    axes[0].set_title("Accuracy")

    sns.lineplot(data=df.loc[:, ["train loss", "test loss"]], ax=axes[1], markers=True)
    axes[1].set_title("Loss")

    # grid
    for ax in axes.flatten():
        ax.grid()


    plt.tight_layout()
    plt.savefig("./Plots/AccuracyLoss.png")
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")