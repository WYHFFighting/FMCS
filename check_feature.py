import numpy as np
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


def cosine_similarity(matrix1, matrix2):
    dot_product = np.dot(matrix1.flatten(), matrix2.flatten())
    norm_a = np.linalg.norm(matrix1)
    norm_b = np.linalg.norm(matrix2)
    return dot_product / (norm_a * norm_b)

# embedding_path = './pretrain_result/ACM_RetainU_NoCommonLinkLoss_CorrLossCoefBeta04/'
embedding_path = r'E:\wyh\论文\安泰\四月\DMG\pretrain_result\higgs\test_split_batch\higgs_com0.npy'
res = np.load(embedding_path)
# embedding_path = open('embedding_path.txt', 'r').read()
# embedding_path = './pretrain_result/ACM_RetainU_NoCommonLinkLoss_CorrLossCoefBeta04_PriMatErr_LinkLosCoef010/'
print(embedding_path)
pri0 = np.load(f'{embedding_path}acm_pri0.npy')
pri1 = np.load(f'{embedding_path}acm_pri1.npy')
com0 = np.load(f'{embedding_path}acm_com0.npy')
com1 = np.load(f'{embedding_path}acm_com1.npy')
acm_data = sio.loadmat(r'./data/acm.mat')
feature = acm_data['feature']
print(cosine_similarity(pri0, com0))
print(cosine_similarity(pri1, com1))
print()
exit()





def plot_heatmap(data, name):
    plt.figure()  # 设置图形的大小
    plt.imshow(data, aspect = 'auto', cmap = 'viridis')  # 使用'viridis'色彩图，也可以选择其他的色彩图，如'hot', 'cool', 'magma'等
    plt.colorbar()  # 显示色彩条
    plt.title(f'{name} Heatmap Visualization')  # 设置图形标题
    plt.xlabel('Feature Index')  # 设置X轴标签
    plt.ylabel('Sample Index')  # 设置Y轴标签
    if not os.path.exists('./figs'):
        os.makedirs('./figs')
    plt.savefig('./figs/{}.pdf'.format(name))
    plt.show()  # 显示图形

# 调用函数绘制热力图
plot_heatmap(pri0, 'pri0')
plot_heatmap(pri1, 'pri1')
plot_heatmap(com0, 'com0')
plot_heatmap(com1, 'com1')
print()



