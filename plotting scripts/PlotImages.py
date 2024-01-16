from LoadDataframe import *
from matplotlib import pyplot as plt

def main():

    configs = ["init_content", "init_random"]

    for config in configs:
        log_dir = f"../logs/{config}"

        df = load_dataframe(log_dir)
    
        num_imgs = df.shape[0]

        for i in range(num_imgs):
            img = df.loc[i, "Image"]

            plt.imshow(img)
            plt.tight_layout()
            plt.axis("off")
            plt.savefig(f"../plots/init_random/{i}.png")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")