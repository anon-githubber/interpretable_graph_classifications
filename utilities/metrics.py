import numpy
import torch
from utilities.fidelity import measure_fidelity
from utilities.contrastivity import measure_contrastivity
from copy import deepcopy
from sklearn import metrics


def auc_scores(all_targets, all_scores):
    all_scores = torch.cat(all_scores).cpu().numpy()
    number_of_classes = int(all_scores.shape[1])

    # For binary classification:
    if number_of_classes == 2:
        # Take only second column (i.e. scores for positive label)
        all_scores = all_scores[:, 1]
        roc_auc = metrics.roc_auc_score(
            all_targets, all_scores, average='macro')
        prc_auc = metrics.average_precision_score(
            all_targets, all_scores, average='macro', pos_label=1)
    # For multi-class classification:
    if number_of_classes > 2:
        # Hand & Till (2001) implementation (ovo)
        roc_auc = metrics.roc_auc_score(
            all_targets, all_scores, multi_class='ovo', average='macro')

        # Todo: build PRC-AUC calculations for multi-class datasets
        prc_auc = "N/A"

    return roc_auc, prc_auc


def compute_metric(trained_classifier_model, GNNGraph_list, metric_attribution_score, dataset_features,config, importance_ranges, cuda):
    fidelity_metric = get_fidelity(trained_classifier_model, metric_attribution_score, GNNGraph_list)


def get_accuracy(trained_classifier_model, GNNgraph_list):
    trained_classifier_model.eval()
    true_pred_pairs = []
    # Instead of sending the whole list as batch, do it one by one in case classifier do not support batch-processing
    for GNNgraph in GNNgraph_list:
        node_feat, n2n, subg = graph_to_tensor(
            GNNgraph, dataset_features["feat_dim"],
            dataset_features["edge_feat_dim"], cmd_args.cuda)

        subg = subg.size()[0]

        output = trained_classifier_model(node_feat, n2n, subg, batch_graph)
        logits = F.log_softmax(output, dim=1)
        pred = logits.data.max(1, keepdim=True)[1]

        print(pred)
        exit()

def get_fidelity(trained_classifier_model, GNNgraph_list, metric_attribution_scores):
    accuracy_prior_occlusion = get_accuracy(trained_classifier_model, GNNgraph_list)
