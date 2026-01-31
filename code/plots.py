import matplotlib.pyplot as plt

def plot_accuracy(df, dataset, perturbation):
    sub = df[(df.dataset == dataset) & (df.perturbation == perturbation)]
    for model in sub.model.unique():
        m = sub[sub.model == model]
        plt.errorbar(m.p, m.acc_mean, yerr=m.acc_std, label=model, marker='o')
    plt.legend()
    plt.xlabel("p")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()
