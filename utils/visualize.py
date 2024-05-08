import torch
import itertools
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from datasets.tep_dataset import TEPDataset
from torch.utils.data.dataloader import DataLoader


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


def plot_embedding(data, label, method='t-SNE', title='t-SNE') -> plt.Figure:
    print('Plotting embeddings...')
    if method == 'pca':
        dim_reduction_tool = PCA(n_components=2, random_state=0)
        title = 'pca'
    else:
        dim_reduction_tool = TSNE(n_components=2, init='pca', random_state=0)
    data = dim_reduction_tool.fit_transform(data)

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    # Min-max 归一化
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.title(title)
    return fig


if __name__ == '__main__':
    model = torch.load(r'F:\StudyFiles\PyProjects\AnomalyDiagnosis\checkpoints\best_model_conv-former.pth')
    dataset = TEPDataset(src_path=r'../data/TEP',
                         split_ratio={'train': 0.7, 'eval': 0.3},
                         data_domains={'source': 1, 'target': 3},
                         dataset_mode='test',
                         data_dim=3,
                         transform=None,
                         overlap=False)
    data_iter = DataLoader(dataset, batch_size=128, shuffle=False)
    for x, y in data_iter:
        if torch.cuda.is_available():
            x = x.to(torch.device('cuda'))
        logit = model(x).to(torch.device('cuda'))
        logit = logit.detach().cpu().numpy()
        logit = np.argmax(logit, axis=1)
        print(logit.shape)

        labels = []
        for item in y:
            labels.extend([item] * x.shape[1])
        x = np.array(x.cpu()).reshape(-1, x.shape[-1])
        labels = np.array(labels)
        fig_prior = plot_embedding(x, labels)
        plt.show()

        labels = []
        for item in logit:
            labels.extend([item] * x.shape[1])
        fig_poster = plot_embedding(x, labels)
        plt.show()
        break
