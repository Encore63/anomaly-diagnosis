import itertools
import numpy as np

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues) -> None:
    """
    this function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'.
    input
    - cm: 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize: true:显示百分比， false: 显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")
    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor('white')

    #
    thresh = cm.max() / 2
    print("thresh", thresh.dtype)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if float(num) > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_embedding(data, label=None, num_classes=10, perplexity=30.0, title='t-SNE') -> plt.Figure:
    print('Plotting embeddings ...')
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=perplexity)
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)
    data = tsne.fit_transform(data)

    # Min-max Scaler
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    if label is None:
        import warnings
        warnings.filterwarnings("ignore")

        kmeans = KMeans(n_clusters=num_classes, random_state=0, n_init=10)
        label = kmeans.fit_predict(data)
        label = np.array(label, dtype=np.float32)

    size = np.ones_like(label) * 100

    fig = plt.figure(figsize=(10, 7))
    ax = plt.subplot(111)
    plt.scatter(data[:, 0], data[:, 1], s=size,
                color=plt.cm.Set3(label),
                marker='o')
    plt.title(title)
    return fig
