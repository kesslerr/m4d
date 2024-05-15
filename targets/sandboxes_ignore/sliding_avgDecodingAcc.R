
# extract average decoding accuracy after baseline in sliding window approaches
# 

#data <- tar_read(data_sliding) %>% filter(experiment=="ERN") %>% filter(times <= -0.6)

data <- tar_read(data_sliding)

experiments <- c("ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3")

baseline_end = c(-0.2, -0.4, 0., 0., 0., 0., 0.) # TODO write paper: in LRP the baselines end at different timepoints, therefore the decoding windows are different for 200ms and 400ms, Here, I chose to equalize the AvgAccuracy esimation to keep it fair 
names(baseline_end) <- c("ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3")

new_data = data.frame()
for (exp in experiments) {
  data_tmp <- data %>% 
    filter(experiment == exp) %>%
    filter(times >= baseline_end[exp])
  avg_accuracy <- data_tmp %>%
    group_by(emc, mac, lpf, hpf, ref, base, det, ar) %>%
    summarize(accuracy = mean(`balanced accuracy`)) %>%
    # reorder columns with accuracy on first place
    select(accuracy, everything()) %>%
    mutate(experiment = exp)
  new_data <- rbind(new_data, avg_accuracy)
}


# 
# 
# 
# # 
# baseline_windows = {
# '200ms': { # correspond to Kappenmann et al.
#   'ERN':  (-.4, -.2), 
#   'LRP':  (-.8, -.6),
#   'MMN':  (-.2, 0.),
#   'N170': (-.2, 0.),
#   'N2pc': (-.2, 0.),
#   'N400': (-.2, 0.),
#   'P3':   (-.2, 0.),
#   'MIPDB': [-.8, -.6],
#   'RSVP': [-.2, 0.],
# },
# '400ms': {
#   'ERN':  (-.6, -.2),
#   'LRP':  (-.8, -.4),
#   'MMN':  (-.4, 0.),
#   'N170': (-.4, 0.),
#   'N2pc': (-.4, 0.),
#   'N400': (-.4, 0.),
#   'P3':   (-.4, 0.),
#   'MIPDB': [-.8, -.4],
#   'RSVP': [-.4, 0.],
# },
# }    