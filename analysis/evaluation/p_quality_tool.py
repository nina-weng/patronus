'''
The code is mainly from InfoDiffusion: https://github.com/isjakewong/InfoDiffusion
A lot thanks to their open source effort.
'''

from global_config import *
import numpy as np
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import scipy
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


""" Impementation of the DCI metric is from:
    https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/dci.py
"""
def compute_dci(mus_train, ys_train, mus_test, ys_test):
      """Computes score based on both training and testing codes and factors."""
      scores = {}
      importance_matrix, train_err, test_err = compute_importance_gbt(
          mus_train, ys_train, mus_test, ys_test)
      assert importance_matrix.shape[0] == mus_train.shape[0]
      assert importance_matrix.shape[1] == ys_train.shape[0]
      scores["informativeness_train"] = train_err
      scores["informativeness_test"] = test_err
      scores['importance'] = importance_matrix
      scores["disentanglement"] = disentanglement(importance_matrix)
      scores["completeness"] = completeness(importance_matrix)
      return scores

def compute_importance_gbt(x_train, y_train, x_test, y_test):
    """Compute importance based on gradient boosted trees."""
    num_factors = y_train.shape[0]
    num_codes = x_train.shape[0]
    importance_matrix = np.zeros(shape=[num_codes, num_factors], dtype=np.float64)
    train_loss = []
    test_loss = []
    for i in range(num_factors):
        model = GradientBoostingClassifier()
        model.fit(x_train.T, y_train[i, :])
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
        test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
    return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11, base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return np.sum(per_code*code_importance)


def completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])


def completeness(importance_matrix):
    """"Compute completeness of the representation."""
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor*factor_importance)


class PredMetric():
    """ Impementation to calculate the AUROC for predicting each attribute
    """
    def __init__(self, predictor = "RandomForest", output_type = "b", attr_names = None, *args, **kwargs):
        super(PredMetric, self).__init__(*args, **kwargs)

        self.attr_names = attr_names
        self._predictor = predictor
        self.output_type = output_type
        if predictor == "Linear":
            self.predictor_class = LogisticRegression
            self.params = {}
            # weights
            self.importances_attr = "coef_"
        elif predictor == "RandomForest":
            self.predictor_class = RandomForestClassifier
            self.importances_attr = "feature_importances_"
            self.params = {"oob_score": True}
        else:
            raise NotImplementedError()

        self.TINY = 1e-12

    def evaluate(self, train_codes, train_attrs, test_codes, test_attrs):
        R = []
        results = []
        # train_codes, test_codes, train_attrs, test_attrs = train_test_split(codes, attrs, test_size=0.2)
        print("Calculate for attribute:")
        for j in range(train_attrs.shape[-1]):
            if isinstance(self.params, dict):
              predictor = self.predictor_class(**self.params)
            elif isinstance(self.params, list):
              predictor = self.predictor_class(**self.params[j])
            else:
              raise NotImplementedError()
            predictor.fit(train_codes, train_attrs[:, j])

            r = getattr(predictor, self.importances_attr)[:, None]
            R.append(np.abs(r))
            # extract relative importance of each code variable in
            # predicting the j attribute
            if self.output_type == "b":
                test_pred_prob = predictor.predict_proba(test_codes)[:, 1]
                tmp_result = roc_auc_score(test_attrs[:, j], test_pred_prob)
            elif self.output_type == "c":
                test_pred = predictor.predict(test_codes)
                tmp_result = accuracy_score(test_attrs[:, j], test_pred)
            results.append(tmp_result)
            if self.attr_names is not None:
                print(j, self.attr_names[j], tmp_result)
            else:
                print(j, tmp_result)

        # R = np.hstack(R) #columnwise, predictions of each z
        results = np.array(results)

        return {
            "{}_avg_result".format(self._predictor): results.mean(),
            "{}_result".format(self._predictor): results
            }

# function that takes a lists of latent indices, thresholds, and signs for classification
class LatentClass(object):
    def __init__(self, targ_ind, lat_ind, is_pos, thresh, __max, __min):
        super(LatentClass, self).__init__()
        self.targ_ind = targ_ind
        self.lat_ind = lat_ind
        self.is_pos = is_pos
        self.thresh = thresh
        self._max = __max
        self._min = __min
        self.it = list(zip(self.targ_ind, self.lat_ind, self.is_pos, self.thresh))

    def __call__(self, z, y_dim):
        # expect z to be [batch, z_dim]
        out = torch.ones((z.shape[0], y_dim))
        for t_i, l_i, is_pos, t in self.it:
            ma, mi = self._max[l_i], self._min[l_i]
            thr = t * (ma - mi) + mi
            res = (z[:, l_i] >= thr if is_pos else z[:, l_i] < thr).type(torch.int)
            out[:, t_i] = res
        return out

class TADMetric():
    """ Impementation of the metric in:
        NashAE: Disentangling Representations Through Adversarial Covariance Minimization
        The code is from:
        https://github.com/ericyeats/nashae-beamsynthesis
    """
    def __init__(self, y_dim, all_attrs):
        self.y_dim = y_dim
        self.all_attrs = all_attrs

    def calculate_auroc(self, targ, targ_ind, lat_ind, z, _ma, _mi, stepsize=0.01):
        thr = torch.arange(0.0, 1.0001, step=stepsize)
        total = targ.shape[0]
        pos_total = targ.sum(dim=0)[targ_ind].item()
        neg_total = total - pos_total
        p_fpr_tpr = torch.zeros((thr.shape[0], 2))
        n_fpr_tpr = torch.zeros((thr.shape[0], 2))
        for i, t in enumerate(thr):
            local_lc = LatentClass([targ_ind], [lat_ind], [True], [t], _ma, _mi)
            pred = local_lc(z.clone(), self.y_dim).to(targ.device)
            p_tp = torch.logical_and(pred == targ, pred).sum(dim=0)[targ_ind].item()
            p_fp = torch.logical_and(pred != targ, pred).sum(dim=0)[targ_ind].item()
            p_fpr_tpr[i][0] = p_fp / neg_total
            p_fpr_tpr[i][1] = p_tp / pos_total
            local_lc = LatentClass([targ_ind], [lat_ind], [False], [t], _ma, _mi)
            pred = local_lc(z.clone(), self.y_dim).to(targ.device)
            n_tp = torch.logical_and(pred == targ, pred).sum(dim=0)[targ_ind].item()
            n_fp = torch.logical_and(pred != targ, pred).sum(dim=0)[targ_ind].item()
            n_fpr_tpr[i][0] = n_fp / neg_total
            n_fpr_tpr[i][1] = n_tp / pos_total
        p_fpr_tpr = p_fpr_tpr.sort(dim=0)[0]
        n_fpr_tpr = n_fpr_tpr.sort(dim=0)[0]
        p_dists = p_fpr_tpr[1:, 0] - p_fpr_tpr[:-1, 0]
        p_area = (p_fpr_tpr[1:, 1] * p_dists).sum().item()
        n_dists = n_fpr_tpr[1:, 0] - n_fpr_tpr[:-1, 0]
        n_area = (n_fpr_tpr[1:, 1] * n_dists).sum().item()
        return p_area, n_area

    def aurocs(self, _z, targ, targ_ind, _ma, _mi):
        # perform a grid search of lat_ind to find the best classification metric
        aurocs = torch.ones(_z.shape[1]) * 0.5  # initialize as random guess
        for lat_ind in range(_z.shape[1]):
            if _ma[lat_ind] - _mi[lat_ind] > 0.2:
                p_auroc, n_auroc = self.calculate_auroc(targ, targ_ind, lat_ind, _z.clone(), _ma, _mi)
                m_auroc = max(p_auroc, n_auroc)
                aurocs[lat_ind] = m_auroc
                # print("{}\t{:1.3f}".format(lat_ind, m_auroc))
        return aurocs

    def aurocs_search(self, a, y):
        aurocs_all = torch.ones((y.shape[1], a.shape[1])) * 0.5
        base_rates_all = y.sum(dim=0)
        base_rates_all = base_rates_all / y.shape[0]
        _ma = a.max(dim=0)[0]
        _mi = a.min(dim=0)[0]
        print("Calculate for attribute:")
        for i in range(y.shape[1]):
            print(i)
            for j in range(a.shape[1]):
                aurocs_all[i, j] = max(roc_auc_score(y.numpy()[:, i], a.numpy()[:, j]), roc_auc_score(y.numpy()[:, i], -a.numpy()[:, j]))
            # aurocs_all[i] = self.aurocs(a, y, i, _ma, _mi)
        return aurocs_all.cpu(), base_rates_all.cpu()

    def evaluate(self, a, y):
        auroc_result, base_rates_raw = self.aurocs_search(torch.FloatTensor(a), torch.IntTensor(y))
        base_rates = base_rates_raw.where(base_rates_raw <= 0.5, 1. - base_rates_raw)
        targ = torch.IntTensor(y)
        dim_y = y.shape[1]

        thresh = 0.75

        ent_red_thresh = 0.2
        max_aur, argmax_aur = torch.max(auroc_result.clone(), dim=1)
        norm_diffs = torch.zeros(dim_y)
        aurs_diffs = torch.zeros(dim_y)
        for ind, tag, max_a, argmax_a, aurs in zip(range(dim_y), self.all_attrs, max_aur.clone(), argmax_aur.clone(),
                                                   auroc_result.clone()):
            norm_aurs = (aurs.clone() - 0.5) / (aurs.clone()[argmax_a] - 0.5)
            aurs_next = aurs.clone()
            aurs_next[argmax_a] = 0.0
            aurs_diff = max_a - aurs_next.max()
            aurs_diffs[ind] = aurs_diff
            norm_aurs[argmax_a] = 0.0
            norm_diff = 1. - norm_aurs.max()
            norm_diffs[ind] = norm_diff

        # calculate mutual information shared between attributes
        # determine which share a lot of information with each other
        with torch.no_grad():
            not_targ = 1 - targ
            j_prob = lambda x, y: torch.logical_and(x, y).sum() / x.numel()
            mi = lambda jp, px, py: 0. if jp == 0. or px == 0. or py == 0. else jp * torch.log(jp / (px * py))

            # Compute the Mutual Information (MI) between the labels
            mi_mat = torch.zeros((dim_y, dim_y))
            for i in range(dim_y):
                # get the marginal of i
                i_mp = targ[:, i].sum() / targ.shape[0]
                for j in range(dim_y):
                    j_mp = targ[:, j].sum() / targ.shape[0]
                    # get the joint probabilities of FF, FT, TF, TT
                    # FF
                    jp = j_prob(not_targ[:, i], not_targ[:, j])
                    pi = 1. - i_mp
                    pj = 1. - j_mp
                    mi_mat[i][j] += mi(jp, pi, pj)
                    # FT
                    jp = j_prob(not_targ[:, i], targ[:, j])
                    pi = 1. - i_mp
                    pj = j_mp
                    mi_mat[i][j] += mi(jp, pi, pj)
                    # TF
                    jp = j_prob(targ[:, i], not_targ[:, j])
                    pi = i_mp
                    pj = 1. - j_mp
                    mi_mat[i][j] += mi(jp, pi, pj)
                    # TT
                    jp = j_prob(targ[:, i], targ[:, j])
                    pi = i_mp
                    pj = j_mp
                    mi_mat[i][j] += mi(jp, pi, pj)

            mi_maxes, mi_inds = (mi_mat * (1 - torch.eye(dim_y))).max(dim=1)
            ent_red_prop = 1. - (mi_mat.diag() - mi_maxes) / mi_mat.diag()

        # calculate Average Norm AUROC Diff when best detector score is at a certain threshold
        filt = (max_aur >= thresh).logical_and(ent_red_prop <= ent_red_thresh)
        aurs_diffs_filt = aurs_diffs[filt]
        max_aur_filt = max_aur[filt]
        # Extract captured attributes directly
        captured_indices = torch.where(filt)[0].tolist()  # Get indices of captured attributes
        captured_attributes = [self.all_attrs[i] for i in captured_indices]  # Map indices to attribute names
        captured_z_dims = argmax_aur[filt].tolist()  # Get z dimensions corresponding to captured attributes
        print(f"Captured Attributes: {captured_attributes}")  # Optional: Print for debugging
        print(f"Captured Attributes acuroc: {max_aur_filt}")
        print(f"Captured z Dimensions: {captured_z_dims}")
        

        return aurs_diffs_filt.sum().item(), auroc_result.cpu().numpy(), len(aurs_diffs_filt)


def eval_disentanglement(ds_name,
                         version_num,
                         return_auroc=False,
                         ):
    '''
    following infoDiffusion. Their code repo: https://github.com/isjakewong/InfoDiffusion
    '''
    npz_pth = REPO_HOME_DIR + f'records/save_latents/{ds_name}-{version_num}/{ds_name}_{version_num}_latent.npz'
    try:
        data_dict = np.load(npz_pth, allow_pickle=True)
    except FileNotFoundError:
        print(f"File not found: {npz_pth}")
        return
    
    print('load npz from {}'.format(npz_pth))


    if  "celeba" in ds_name.lower() :
        y_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
                   'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
                   'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                   'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                   'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                   'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
                   'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                   'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
                   ]
        output_type = "b"
    elif ds_name.lower() == "fmnist":
        y_names = ["Class"]
        output_type = "c"
    elif ds_name.lower() == "cifar10":
        y_names = ["Class"]
        output_type = "c"
    elif ds_name.lower() == "ffhq":
        y_names = ["Age", "Gender", "Glass"]
        output_type = "c"
    elif ds_name.lower() == "chexpert":
        # it has to be the same as in `proto_ddpm/dataloader.py`
        y_names = [# disease
                'No Finding', 'Enlarged Cardiomediastinum',
                'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
                'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                'Pleural Other', 'Fracture', 'Support Devices',
                # demographic info
                'age_group_b','sex_b', 'race_b', 'ethnicity_b', 
                # other info
                'insurance_type_b','interpreter_needed_b', 
                'deceased_b','bmi_group_b',
                'PM']
        output_type = "b"

    if  "celeba" in ds_name.lower() :
        a = data_dict["all_a"][:10000,:]
        y = data_dict["all_attr"][:10000, :].astype(np.int64)  # Or np.int32, depending on your needs

    elif ds_name.lower() == "ffhq":
        a = data_dict["all_a"][:10000,:]
        y = data_dict["all_attr"][:10000, :].astype(np.int64)  # Or np.int32, depending on your needs
    elif ds_name.lower() == "chexpert":
        a = data_dict["all_a"][:10000,:]
        y = data_dict["all_attr"][:10000, :].astype(np.int64)  # Or np.int32, depending on your needs
    else:
        a = data_dict["all_a"]
        if len(data_dict["all_attr"].shape) == 2:
            y = data_dict["all_attr"][:, :].astype(np.int)
        else:
            # y = data_dict["all_attr"][:, np.newaxis].astype(np.int)
            y = data_dict["all_attr"][:, np.newaxis].astype(np.int64)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    preds_rf, avg_preds_rf = [], []
    preds_ln, avg_preds_ln = [], []
    if "celeba" in ds_name.lower() or ds_name.lower() == "chexpert":
        tad_scores, tad_attrs = [], []

    auroc_result_all = []

    for tr_idx, te_idx in kf.split(a):
        tr_a, te_a = a[tr_idx], a[te_idx]
        tr_y, te_y = y[tr_idx], y[te_idx]
        std = StandardScaler()
        std.fit(tr_a)
        tr_a = std.transform(tr_a)
        te_a = std.transform(te_a)

        if  "celeba" in ds_name.lower() or ds_name.lower() == "chexpert":
            tad_metric = TADMetric(y.shape[1], y_names)
            tad_score, auroc_result, num_attr = tad_metric.evaluate(tr_a, tr_y)
            auroc_result_all.append(auroc_result)
            #
            print("TAD SCORE: ", tad_score, "Attributes Captured: ", num_attr)
            tad_scores.append(tad_score)
            tad_attrs.append(num_attr)

    

        pred_metric = PredMetric("Linear", output_type, y_names)
        pred_result = pred_metric.evaluate(tr_a, tr_y, te_a, te_y)

        print("Avg Result", pred_result['Linear_avg_result'])
        avg_preds_ln.append(pred_result['Linear_avg_result'])
        preds_ln.append(pred_result['Linear_result'])


    if ds_name.lower() == "celeba" or ds_name.lower() == "chexpert":
        tad_scores = np.array(tad_scores)
        tad_attrs = np.array(tad_attrs)
        print("TAD Score, {:.4f} \pm {:.4f}".format(np.array(tad_scores).mean(), np.array(tad_scores).std()))
        print("TAD Attr, {:.4f} \pm {:.4f}".format(np.array(tad_attrs).mean(), np.array(tad_attrs).std()))

    avg_preds_ln = np.array(avg_preds_ln)
    print("Avg Acc (Linear), {:.4f} \pm {:.4f}".format(np.array(avg_preds_ln).mean(), np.array(avg_preds_ln).std()))

    preds_ln = np.vstack(preds_ln)
    for a_idx in range(preds_ln.shape[1]):
        print("Acc for {} (Linear), {:.4f} \pm {:.4f}".format(y_names[a_idx], preds_ln[:, a_idx].mean(), preds_ln[:, a_idx].std()))



    if return_auroc == True:
        return auroc_result_all, y_names
    else:
        return None
    
