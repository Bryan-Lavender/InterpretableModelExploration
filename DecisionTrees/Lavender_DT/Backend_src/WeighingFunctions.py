import numpy as np

#Known as Variance Weighting
def Variance_Weighted_sum(FI, out_logits):
    output_variance = np.var(out_logits, axis=0)
    norm_var = (output_variance - output_variance.min()) / (output_variance.max() - output_variance.min())
    mean_fi = FI.mean(axis=0)
    weighted = mean_fi * norm_var[:, np.newaxis]  # shape: (n_out, n_in)
    result = weighted.sum(axis=0)
    return result[np.newaxis, :][0]

#Known as Max_All
def Max_Over_Outcome(FI, out_logits=None):
    mean_fi = FI.mean(axis=0)  
    max_fi = mean_fi.max(axis=0) 
    return max_fi[np.newaxis, :][0]

#Known as Max_Average
def Max_Average_FI(FI, out_logits=None):
    mean_fi = FI.mean(axis=0)
    output_avgs = mean_fi.mean(axis=1)
    best_index = np.argmax(output_avgs)
    selected_row = mean_fi[best_index]  
    return selected_row[np.newaxis, :][0]

def Double_Average(FI, out_logits=None):
    mean_per_output = FI.mean(axis=0)
    double_avg = mean_per_output.mean(axis=0)
    return double_avg[np.newaxis, :][0]

def Classification(FI, out_logits):
    argmax_indices = np.argmax(out_logits, axis=1)  
    n = FI.shape[0]
    selected = FI[np.arange(n), argmax_indices]
    return np.mean(selected, axis = 0)