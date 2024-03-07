
# plot ecdf with colored dots for top pipelines



data <- tar_read(data_tsum)

best_data = data.frame()
for (experiment_val in c("ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3")){
   newdata <- data %>%
    #group_by(ref, hpf, lpf, emc, mac, base, det, ar) %>%
    #summarize(tsum = mean(tsum)) %>%
    filter(experiment==experiment_val) %>%
    mutate(performance = 
             (ar==FALSE) & 
             #ref=="P9P10"
             #base=="400ms" & 
             (det=="linear") & 
             (emc=="None") & 
             (mac=="None") & 
             (hpf==0.5) & 
             (lpf==6)
    ) %>% # if these conditions are met, then write TRUE, else FALSE
    #ungroup() %>% # because row names etc are not correct in a grouped df
    arrange(tsum) %>% # sort by ascending tsum (if not, write desc(tsum))
    mutate(idx = as.numeric(row.names(.))/1152) # index column
  best_data = rbind(best_data, newdata)
}

#ggplot(best_data, aes(x=tsum, fill=performance)) +
#  geom_histogram()


ggplot(best_data, aes(x=tsum, # index as x
                      #y=idx
                      )) +
  geom_hline(data =filter(best_data, performance==TRUE),
             aes(yintercept = idx),
             color="darkgrey",
             ) +
  #geom_point(size=1) +
  stat_ecdf() +
  facet_wrap(experiment ~., scales = "free_x")

# single pipeline data makes only sence for itself, not in a plot with all signle experiment, because some experiments have lower tsum, some higher...