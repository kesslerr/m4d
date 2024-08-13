
# quantify for each fp, how many participants dropped out
# TODO: EEGNet and Sliding separately
# TODO for each experiment

data <- tar_read(data_eegnet)
experiments = unique(data$experiment)

# DEBUG
experiment <- experiments[1]

data_exp <- data[data$experiment == experiment,]

# make fp a variable as concatenation of emc   mac   lpf   hpf   ref     det    base  ar
data_exp$fp <- paste(data_exp$emc, data_exp$mac, data_exp$lpf, data_exp$hpf, data_exp$ref, data_exp$det, data_exp$base, data_exp$ar, sep="_")

# group by fp, count subjects
res <- data_exp %>% group_by(c( "emc","mac","lpf","hpf","ref","det","base","ar")) %>% summarise(n = n())

# order by n ascending
res <- res[order(res$n),]