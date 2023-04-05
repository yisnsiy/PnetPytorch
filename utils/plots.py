# from os.path import join
# import numpy as np
# import itertools
# from matplotlib import pyplot as plt
# from sklearn import metrics
#
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.gcf().subplots_adjust(bottom=0.25)
#
# def save_confusion_matrix(cnf_matrix, saving_dir, model):
#     plt.figure()
#     plot_confusion_matrix(cnf_matrix, classes=[0, 1],
#                           title='Confusion matrix, without normalization')
#     file_name = join(saving_dir, 'confusion_' + model + '.png')
#     plt.savefig(file_name)
#     # plt.show()
#
#     plt.figure()
#     plot_confusion_matrix(cnf_matrix, normalize=True, classes=[0, 1],
#                           title='Normalized confusion matrix')
#     file_name = join(saving_dir, 'confusion_normalized_' + model)
#     plt.savefig(file_name)
#     # plt.show()
#
# def save_metrics(effect, saving_dir):
#     plt.figure()
#     if type(effect).__name__ == 'dict':
#         ax = list(effect.keys())
#         ay = list(effect.values())
#         plt.ylim([0.0, 1.05])
#         plt.tick_params(axis='x', labelsize=12)
#         plt.bar(ax, ay)
#         for a,b,i in zip(ax,ay,range(len(ax))): # zip 函数
#             plt.text(a,b+0.01,"%.2f"%ay[i],ha='center',fontsize=12)
#         # plt.show()
#         plt.savefig(join(saving_dir, "metics"))
#
# def plot_roc(fig, labels, preScore, save_dir, label=''):
#     fpr, tpr, thresholds = metrics.roc_curve(labels, preScore, pos_label=1)
#     roc_auc = metrics.auc(fpr, tpr)
#     plt.figure(fig.number)
#     plt.plot(fpr, tpr, label=label + ' (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate', fontsize=12)
#     plt.ylabel('True Positive Rate', fontsize=12)
#     plt.title('Receiver operating characteristic (ROC)', fontsize=12)
#     plt.legend(loc="lower right")
#
# def plot_prc(fig, labels, preScore, save_dir, label=''):
#     precision, recall, thresholds = metrics.precision_recall_curve(labels, preScore, pos_label=1)
#     roc_auc = metrics.average_precision_score(labels, preScore)
#     plt.figure(fig.number)
#     plt.plot(recall, precision, label=label + ' (area under precision recall curve = %0.2f)' % roc_auc)
#
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('Recall', fontsize=12)
#     plt.ylabel('precision', fontsize=12)
#     plt.title('Precisoin Recall Curve (PRC)', fontsize=12)
#     plt.legend(loc="lower right")
#
# def save_auc_auprc_curve(preScore, labels, saving_dir, model_name):
#     auc_fig = plt.figure()
#     auc_fig.set_size_inches((10, 6))
#     prc_fig = plt.figure()
#     prc_fig.set_size_inches((10, 6))
#
#     plot_roc(auc_fig, labels, preScore, saving_dir, label=model_name)
#     plot_prc(prc_fig, labels, preScore, saving_dir, label=model_name)
#
#     # auc_fig.show()
#     # prc_fig.show()
#     auc_fig.savefig(join(saving_dir, 'auc_curves'))
#     prc_fig.savefig(join(saving_dir, 'auprc_curves'))