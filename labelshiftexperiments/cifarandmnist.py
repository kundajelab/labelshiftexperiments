import glob
import numpy as np
from collections import defaultdict, OrderedDict
import scipy
import sys
import abstention.calibration


def read_labels(fh):
    to_return = []
    for line in fh:
        the_class=int(line.rstrip())
        to_add = np.zeros(10)
        to_add[the_class] = 1
        to_return.append(to_add)
    return np.array(to_return)


def read_preds(fh):
    return np.array([[float(x) for x in y.rstrip().split("\t")]
                      for y in fh])


def sample_from_probs_arr(arr_with_probs, rng):
    rand_num = rng.uniform()
    cdf_so_far = 0
    for (idx, prob) in enumerate(arr_with_probs):
        cdf_so_far += prob
        if (cdf_so_far >= rand_num
            or idx == (len(arr_with_probs) - 1)):  # need the
            # letterIdx==(len(row)-1) clause because of potential floating point errors
            # that mean arrWithProbs doesn't sum to 1
            return idx


def get_func_to_draw_label_proportions(test_labels):
    test_class_to_indices = defaultdict(list)
    for index,row in enumerate(test_labels):
        row_label = np.argmax(row)
        test_class_to_indices[row_label].append(index)
    def draw_test_indices(total_to_return, label_proportions, rng):
        indices_to_use = []
        for class_index, class_proportion in enumerate(label_proportions):
            indices_to_use.extend(rng.choice(
                    test_class_to_indices[class_index],
                    int(total_to_return*class_proportion),
                    replace=True))
        for i in range(total_to_return-len(indices_to_use)):
            class_index = sample_from_probs_arr(label_proportions, rng)
            indices_to_use.append(
                rng.choice(test_class_to_indices[class_index]))
        return indices_to_use
    return draw_test_indices


def run_calibmethods(valid_preacts, valid_labels,
                     test_preacts, calibname_to_calibfactory,
                     samplesize,
                     samplesizesseen,
                     metric_to_samplesize_to_calibname_to_unshiftedvals):
    calibname_to_calibfunc = {}
    calibname_to_calibvalidpreds = {}
    for calibname, calibfactory in\
                                 calibname_to_calibfactory.items():
        calibfunc = calibfactory(
            valid_preacts=valid_preacts,
            valid_labels=valid_labels)

        unshifted_test_preds = calibfunc(test_preacts)
        unshifted_test_nll = -np.mean(
                                np.sum(np.log(unshifted_test_preds)
                                *test_labels, axis=-1))
        unshifted_test_ece = abstention.calibration.compute_ece(
                              softmax_out=unshifted_test_preds,
                              labels=test_labels, bins=15)
        unshifted_test_jsdiv =\
            scipy.spatial.distance.jensenshannon(                       
               p=np.mean(unshifted_test_preds, axis=0),              
               q=np.mean(test_labels, axis=0))

        #if statement is there to avoid double-counting
        if (samplesize not in samplesizesseen):
            metric_to_samplesize_to_calibname_to_unshiftedvals[
                'ece'][samplesize][calibname].append(unshifted_test_ece)
            metric_to_samplesize_to_calibname_to_unshiftedvals[
                'nll'][samplesize][calibname].append(unshifted_test_nll)
            metric_to_samplesize_to_calibname_to_unshiftedvals[
                'jsdiv'][samplesize][calibname].append(unshifted_test_jsdiv)
        calibname_to_calibfunc[calibname] = calibfunc
        calibname_to_calibvalidpreds[calibname] = calibfunc(valid_preacts)

    return (calibname_to_calibfunc, calibname_to_calibvalidpreds)


def run_experiments(num_trials, seeds, alphas_and_samplesize,
                    shifttype,
                    calibname_to_calibfactory,
                    imbalanceadaptername_to_imbalanceadapter,
                    adaptncalib_pairs, 
                    validglobprefix,
                    testglobprefix,
                    valid_labels, test_labels):

    draw_test_indices = get_func_to_draw_label_proportions(test_labels)

    alpha_to_samplesize_to_adaptername_to_metric_to_vals =(
        defaultdict(
            lambda: defaultdict(
                     lambda: defaultdict(
                              lambda: defaultdict(list)))))
    alpha_to_samplesize_to_baselineacc = defaultdict(
        lambda: defaultdict(list))
    metric_to_samplesize_to_calibname_to_unshiftedvals = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))

    samplesizesseen = set()
    for (alpha,samplesize) in alphas_and_samplesize:
        for seed in seeds:
            print("Seed",seed)
            for trial_num in range(num_trials):   
                rng = np.random.RandomState(seed*num_trials + trial_num)
                test_preacts = read_preds(
                    open(glob.glob(testglobprefix+str(seed)+"*.txt")[0]))
                valid_preacts = read_preds(
                    open(glob.glob(validglobprefix+str(seed)+"*.txt")[0]))
                #let's also sample different validation sets
                # according to the random seed AND the trialnum
                sample_valid_indices = rng.choice(
                      a=np.arange(len(valid_preacts)),
                      size=samplesize, replace=False)
                sample_valid_preacts = valid_preacts[sample_valid_indices]
                sample_valid_labels = valid_labels[sample_valid_indices]

                (calibname_to_calibfunc,
                 calibname_to_calibvalidpreds) = (
                    run_calibmethods(
                       valid_preacts=sample_valid_preacts,
                       valid_labels=sample_valid_labels,
                       test_preacts=test_preacts,
                       calibname_to_calibfactory=calibname_to_calibfactory,
                       samplesize=samplesize,
                       samplesizesseen=samplesizesseen,
                       metric_to_samplesize_to_calibname_to_unshiftedvals=
                        metric_to_samplesize_to_calibname_to_unshiftedvals))

                if (shifttype=='dirichlet'):
                    altered_class_priors = rng.dirichlet([
                        alpha for x in range(10)])
                elif (shifttype=='tweakone'):
                    altered_class_priors = np.full((10), (1.0-alpha)/9)
                    altered_class_priors[3] = alpha
                else:
                    raise RuntimeError("Unsupported shift type",shifttype)

                test_indices = draw_test_indices(
                                 total_to_return=samplesize,
                                 label_proportions=altered_class_priors,
                                 rng=rng)
                shifted_test_labels = test_labels[test_indices]
                shifted_test_preacts = test_preacts[test_indices]

                calibname_to_calibshiftedtestpreds = {}
                for (calibname, calibfunc) in calibname_to_calibfunc.items():
                    calibname_to_calibshiftedtestpreds[calibname] =(
                        calibfunc(shifted_test_preacts))

                shifted_test_baseline_accuracy = np.mean(
                    np.argmax(shifted_test_labels,axis=-1)==
                    np.argmax(abstention.calibration.softmax(
                                preact=shifted_test_preacts,
                                temp=1.0, biases=None),axis=-1))
                alpha_to_samplesize_to_baselineacc[alpha][samplesize].append(
                      shifted_test_baseline_accuracy)

                true_shifted_priors = np.mean(shifted_test_labels, axis=0)
                for adapter_name,calib_name in adaptncalib_pairs:
                    calib_shifted_test_preds =\
                      calibname_to_calibshiftedtestpreds[calib_name]
                    calib_valid_preds = calibname_to_calibvalidpreds[
                                         calib_name]
                    imbalance_adapter =\
                      imbalanceadaptername_to_imbalanceadapter[adapter_name]
                    imbalance_adapter_func = imbalance_adapter(
                        valid_labels=sample_valid_labels,
                        tofit_initial_posterior_probs=calib_shifted_test_preds,
                        valid_posterior_probs=calib_valid_preds)
                    shift_weights = imbalance_adapter_func.multipliers
                    adapted_shifted_test_preds = imbalance_adapter_func(
                        calib_shifted_test_preds)
                    estim_shifted_priors = np.mean(adapted_shifted_test_preds,
                                                   axis=0)
                    adapted_shifted_test_accuracy = np.mean(
                        np.argmax(shifted_test_labels,axis=-1)==
                        np.argmax(adapted_shifted_test_preds,axis=-1))
                    delta_from_baseline = (adapted_shifted_test_accuracy
                                           -shifted_test_baseline_accuracy)

                    alpha_to_samplesize_to_adaptername_to_metric_to_vals[
                        alpha][samplesize][adapter_name+":"+calib_name][
                        'jsdiv'].append(
                          scipy.spatial.distance.jensenshannon(
                              p=true_shifted_priors, q=estim_shifted_priors))
                    alpha_to_samplesize_to_adaptername_to_metric_to_vals[
                        alpha][samplesize][adapter_name+":"+calib_name][
                        'delta_acc'].append(delta_from_baseline)
    
        if (samplesize not in samplesizesseen):
            print("Calibration stats")
            for metric in ['ece', 'nll', 'jsdiv']:
                print("Metric",metric)
                for calibname in calibname_to_calibfactory:
                    print(calibname, np.mean(
                           metric_to_samplesize_to_calibname_to_unshiftedvals[
                            metric][samplesize][calibname]))         
            samplesizesseen.add(samplesize)
        
        print("On alpha",alpha,"sample size", samplesize)
        for metric_name in ['delta_acc', 'jsdiv']:
            print("Metric",metric_name)
            for adapter_name,calib_name in adaptncalib_pairs:
                adaptncalib_name = adapter_name+":"+calib_name
                n = len(alpha_to_samplesize_to_adaptername_to_metric_to_vals[
                      alpha][samplesize][adaptncalib_name][metric_name])
                
                print(adaptncalib_name, np.mean(
                 alpha_to_samplesize_to_adaptername_to_metric_to_vals[
                   alpha][samplesize][adaptncalib_name][metric_name]), "+/-",
                 (1.0/np.sqrt(n))*np.std(
                 alpha_to_samplesize_to_adaptername_to_metric_to_vals[
                   alpha][samplesize][adaptncalib_name][metric_name],
                   ddof=1))
                sys.stdout.flush()
    

    return (alpha_to_samplesize_to_adaptername_to_metric_to_vals,
            alpha_to_samplesize_to_baselineacc,
            metric_to_samplesize_to_calibname_to_unshiftedvals)
    
