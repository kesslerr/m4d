res <- tar_read(reults_sliding)

library(ggplot2)

ggplot(data = res, aes(x=level, y=tsum)) +
  geom_bar(stat="identity") + 
  facet_wrap(. ~variable, scales="free_x")


res <- tar_read(reults_sliding_experiment)

ggplot(data = res, aes(x=level, y=tsum)) + #, fill=experiment
  geom_bar(stat="identity", position=position_dodge()) + 
  #scale_fill_grey() +
  #facet_wrap(. ~variable, scales="free_x")
  facet_grid(experiment ~variable, scales = "free_x")



# plot sliding time courses for example df
data <- tar_read(data_sliding)

# filter for 1 fp per experiment
luckfps <- data.frame(
    experiment = c('ERN', 'LRP', 'MMN', 'N170', 'N2pc', 'N400', 'P3'),
    ref = c('P9P10', 'P9P10', 'P9P10', 'average', 'P9P10', 'P9P10', 'P9P10'),
    hpf = c('0.1', '0.1', '0.1', '0.1', '0.1', '0.1', '0.1'),
    lpf = c('None', 'None', 'None', 'None', 'None', 'None', 'None'),
    emc = c('ica', 'ica', 'ica', 'ica', 'ica', 'ica', 'ica'),
    mac = c('ica', 'ica', 'ica', 'ica', 'ica', 'ica', 'ica'),
    base = c('200ms', '200ms', '200ms', '200ms', '200ms', '200ms', '200ms'),
    det = c('offset', 'offset', 'offset', 'offset', 'offset', 'offset', 'offset'),
    ar = c('TRUE', 'TRUE', 'TRUE', 'TRUE', 'TRUE', 'TRUE', 'TRUE')
  )

library(dplyr)
data_fp <- semi_join(data, luckfps, 
                               by = c("experiment", "ref", "hpf", "lpf", "emc", "mac", "base", "det", "ar"))

print(data_fp)


### single --> todo: there seems to be some double shading... couldnt resolve yet


# ggplot(data_fp, aes(x = times, y = `balanced accuracy`)) +
#   geom_line() +
#   geom_hline(yintercept=0.5, linetype="dotted") +
#   geom_rect(data = subset(data_fp, significance == TRUE),
#             aes(xmin = times, 
#                 xmax = lead(times),
#                 ymin = -Inf, 
#                 ymax = Inf),
#             fill = "red",
#             alpha = 0.2) +
#   facet_grid(experiment~., scales = "free_x") 

# dfvlines = data.frame(
#   experiment = c('ERN', 'LRP', 'MMN', 'N170', 'N2pc', 'N400', 'P3'),
#   x =
# )

# with points instead
ggplot(data_fp, aes(x = times, y = `balanced accuracy`)) +
  geom_line() +
  geom_hline(yintercept=0.5, linetype="solid") +
  geom_vline(xintercept=0, linetype="dashed") +
  geom_point(data=filter(data_fp, significance=="TRUE"),
             aes(
                 x=times,
                 y=0.48),
                 color="darkred",
                 size=1,
             ) +
  facet_wrap(experiment~., scales = "free_x", ncol=1) +
  scale_x_continuous(breaks = seq(-8, 8, by = 2)/10, 
                     labels = seq(-8, 8, by = 2)/10) 



