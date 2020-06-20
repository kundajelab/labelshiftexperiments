import numpy as np
from collections import defaultdict
from abstention.figure_making_utils import (
    wilcox_srs, get_ustats_mat,
    get_top_method_indices)

def get_methodname_to_ranks(methodname_to_vals, methodnames, sortsign):
    methodname_to_ranks = defaultdict(list)
    for i in range(len(methodname_to_vals[methodnames[0]])):
        methodname_and_val = [
            (x, methodname_to_vals[x][i]) for x in methodnames]
        rank_and_methodnameandval = enumerate(
            sorted(methodname_and_val, key=lambda x: sortsign*x[1]))
        methodname_and_rank = [(x[1][0], x[0])
                               for x in rank_and_methodnameandval]
        for methodname, rank in methodname_and_rank:
            methodname_to_ranks[methodname].append(rank)
    return methodname_to_ranks


def stderr(vals):
    return (1.0/np.sqrt(len(vals)))*np.std(vals, ddof=1)


def render_calibration_table(
    metric_to_samplesize_to_calibname_to_unshiftedvals,
    ustat_threshold, metrics_in_table,
    samplesizes_in_table, calibnames_in_table,
    metricname_to_nicename, calibname_to_nicename, caption, label,
    applyunderline,
    decimals=3):

    metric_to_samplesize_to_calibname_to_ranks = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    metric_to_samplesize_to_bestmethods = defaultdict(lambda: dict())
    metric_to_samplesize_to_toprankedmethod = defaultdict(lambda: dict())
    
    for metricname in metrics_in_table:
        for samplesize in samplesizes_in_table:
            methodname_to_vals =\
              metric_to_samplesize_to_calibname_to_unshiftedvals[metricname][samplesize]
            methodname_and_avgvals = [
                (methodname, np.median(methodname_to_vals[methodname]))
                 for methodname in calibnames_in_table]
            toprankedmethod = (
                min(methodname_and_avgvals, key=lambda x: x[1])[0]
                if applyunderline else None)
            ustats_mat = get_ustats_mat(
                          method_to_perfs=methodname_to_vals,
                          method_names=calibnames_in_table)
            tied_top_methods = (
                get_top_method_indices(
                    sorting_metric_vals=[x[1] for x in methodname_and_avgvals],
                    ustats_mat=ustats_mat,
                    threshold=ustat_threshold,
                    largerisbetter=False))
            metric_to_samplesize_to_bestmethods[metricname][samplesize] = (
              [calibnames_in_table[x] for x in tied_top_methods])
            metric_to_samplesize_to_calibname_to_ranks[
                metricname][samplesize] = (
                 get_methodname_to_ranks(methodname_to_vals=methodname_to_vals,
                                         methodnames=calibnames_in_table,
                                         sortsign=1))
            metric_to_samplesize_to_toprankedmethod[
                metricname][samplesize] = toprankedmethod
    
    toprint = ("""
\\begin{table*}
\\adjustbox{max width=\\textwidth}{
  \\centering
  \\begin{tabular}{ c | """+" | ".join([" ".join(["c" for samplesize in samplesizes_in_table])
                                         for metricname in metrics_in_table])+""" }
    \\multirow{2}{*}{\\begin{tabular}{c}\\textbf{Calibration} \\\\ \\textbf{Method} \\end{tabular}} & """
    +(" & ".join(["\\multicolumn{"+str(len(samplesizes_in_table))+"}{| c}{"+metricname_to_nicename[metricname]+"}"
                  for metricname in metrics_in_table]))+"""\\\\
    \cline{2-"""+str(1+len(metrics_in_table)*len(samplesizes_in_table))+"""}
    & """+(" & ".join([" & ".join(["$n$="+str(samplesize) for samplesize in samplesizes_in_table])
                         for metricname in metrics_in_table]))+"\\\\\n    \hline\n    "+
"\n    ".join([
    calibname_to_nicename[calibname]+" & "+(" & ".join([
           ("\\textbf{" if calibname in metric_to_samplesize_to_bestmethods[metricname][samplesize] else "")
           +("\\underline{" if calibname==metric_to_samplesize_to_toprankedmethod[metricname][samplesize] else "")
           +str(np.round(np.median(metric_to_samplesize_to_calibname_to_unshiftedvals[metricname][samplesize][calibname]), decimals=decimals))
           #+" +/- "
           #+str(np.round(stderr(metric_to_samplesize_to_calibname_to_unshiftedvals[metricname][samplesize][calibname]), decimals=decimals))
           +"; "
           +str(np.round(np.median(metric_to_samplesize_to_calibname_to_ranks[metricname][samplesize][calibname]), decimals=decimals))
           #+" +/-"
           #+str(np.round(stderr(metric_to_samplesize_to_calibname_to_ranks[metricname][samplesize][calibname]), decimals=decimals))
           +("}" if calibname==metric_to_samplesize_to_toprankedmethod[metricname][samplesize] else "")
           +("}" if calibname in metric_to_samplesize_to_bestmethods[metricname][samplesize] else "")
           for metricname in metrics_in_table for samplesize in samplesizes_in_table
         ]))+"\\\\"
     for calibname in calibnames_in_table
])
+"""
  \\end{tabular}}
  \\caption{"""+caption+"""}
  \\label{tab:"""+label+"""}
\\end{table*}
""")
    return toprint


def render_adaptation_table(
    alpha_to_samplesize_to_adaptncalib_to_metric_to_vals,
    ustat_threshold,
    valmultiplier,
    adaptname_to_nicename, calibname_to_nicename,
    methodgroups, metric, largerisbetter,
    alphas_in_table, samplesizes_in_table, caption, label,
    applyunderline,
    symbol='\\alpha',
    decimals=3):
  
    methodgroupname_to_alpha_to_samplesize_to_bestmethods =\
        defaultdict(lambda: defaultdict(lambda: {}))
    methodgroupname_to_alpha_to_samplesize_to_toprankedmethod =\
        defaultdict(lambda: defaultdict(lambda: {}))
    methodgroupname_to_alpha_to_samplesize_to_methodname_to_ranks =\
        defaultdict(lambda: defaultdict(lambda: {}))
    
    for methodgroupname in methodgroups:
        for alpha in alphas_in_table:
            for samplesize in samplesizes_in_table:
                methodname_to_vals = dict(
                    [(methodname,
                      alpha_to_samplesize_to_adaptncalib_to_metric_to_vals[
                       alpha][samplesize][methodname][metric])
                     for methodname in methodgroups[methodgroupname]])
                methodname_and_avgvals = [
                  (methodname, np.median(methodname_to_vals[methodname]))
                   for methodname in methodgroups[methodgroupname]]
                toprankedmethod = min(methodname_and_avgvals,
                    key=lambda x: (-1 if largerisbetter else 1)*x[1])[0]
                ustats_mat = get_ustats_mat(
                                method_to_perfs=methodname_to_vals,
                                method_names=methodgroups[methodgroupname])
                tied_top_methods = (
                    get_top_method_indices(
                        sorting_metric_vals=[x[1] for x in methodname_and_avgvals],
                        ustats_mat=ustats_mat,
                        threshold=ustat_threshold,
                        largerisbetter=largerisbetter))
                
                methodgroupname_to_alpha_to_samplesize_to_bestmethods[
                  methodgroupname][alpha][samplesize] = (
                    [methodgroups[methodgroupname][x] for x in tied_top_methods])
                methodgroupname_to_alpha_to_samplesize_to_toprankedmethod[
                    methodgroupname][alpha][samplesize] = (
                        toprankedmethod if applyunderline else None) 
                methodgroupname_to_alpha_to_samplesize_to_methodname_to_ranks[
                    methodgroupname][alpha][samplesize] = (
                        get_methodname_to_ranks(
                            methodname_to_vals=methodname_to_vals,
                            methodnames=methodgroups[methodgroupname],
                            sortsign=(-1 if largerisbetter else 1)))
      
    toprint = ("""
\\begin{table*}
\\adjustbox{max width=\\textwidth}{
  \\centering
  \\begin{tabular}{ c | c | """+(" | ".join([ " ".join( ["c" for samplesize in samplesizes_in_table ] ) for alpha in alphas_in_table]))+"}\n")
    toprint += ("    \\multirow{2}{*}{\\begin{tabular}{c}\\textbf{Shift} \\\\ \\textbf{Estimator} \\end{tabular}}"
                +" & \\multirow{2}{*}{\\begin{tabular}{c}\\textbf{Calibration} \\\\ \\textbf{Method} \\end{tabular}} & "
                +((" & ".join(["\\multicolumn{"+str(len(samplesizes_in_table))+"}{| c}{$"+symbol+"="+str(alpha)+"$}"
                              for alpha in alphas_in_table]))+"\\\\ \n")
                +"    \\cline{3-"+str(2+len(alphas_in_table)*len(samplesizes_in_table))+"}\n"
                +"    & & "+(" & ".join([" & ".join(["$n$="+str(samplesize) for samplesize in samplesizes_in_table])
                          for alpha in alphas_in_table]))+"\\\\")
    #toprint += "    \\hline \\hline"
    for methodgroupnum, methodgroupname in enumerate(methodgroups.keys()):
        #if (methodgroupnum > 0):
        toprint += "\n    \\hline\n    \\hline"
        for adaptncalib in methodgroups[methodgroupname]:
            adaptname = adaptncalib.split(":")[0]
            calibname = adaptncalib.split(":")[1]
            toprint += "\n    "
            toprint += adaptname_to_nicename[adaptname]
            toprint += " & "+calibname_to_nicename[calibname]
            toprint += " & "
            toprint += " & ".join([
               ("\\textbf{" if adaptncalib in methodgroupname_to_alpha_to_samplesize_to_bestmethods[methodgroupname][alpha][samplesize] else "")
               +("\\underline{" if adaptncalib==methodgroupname_to_alpha_to_samplesize_to_toprankedmethod[methodgroupname][alpha][samplesize] else "")
               +str(np.round(valmultiplier*np.median(alpha_to_samplesize_to_adaptncalib_to_metric_to_vals[alpha][samplesize][adaptncalib][metric]), decimals=decimals))
               #+" +/- "
               #+str(np.round(stderr(alpha_to_samplesize_to_adaptncalib_to_metric_to_vals[alpha][samplesize][adaptncalib][metric]), decimals=decimals))
               +"; "
               +str(np.round(np.median(methodgroupname_to_alpha_to_samplesize_to_methodname_to_ranks[methodgroupname][alpha][samplesize][adaptncalib]), decimals=decimals))
               #+" +/-"
               #+str(np.round(stderr(methodgroupname_to_alpha_to_samplesize_to_methodname_to_ranks[methodgroupname][alpha][samplesize][adaptncalib]), decimals=decimals))
               +("}" if adaptncalib==methodgroupname_to_alpha_to_samplesize_to_toprankedmethod[methodgroupname][alpha][samplesize] else "")
               +("}" if adaptncalib in methodgroupname_to_alpha_to_samplesize_to_bestmethods[methodgroupname][alpha][samplesize] else "")
               for alpha in alphas_in_table for samplesize in samplesizes_in_table ])
            toprint += "\\\\"
    
    toprint += """
  \\end{tabular}}
  \\caption{"""+caption+"""}
  \\label{tab:"""+label+"""}
\\end{table*}
"""
    return toprint
