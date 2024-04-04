
# sociodem EDA

demo <- tar_read(demographics)

model <- tar_read(eegnet_HLM) # TODO, also branches
orig_data <- tar_read(data_eegnet)

# from rfx_vis function
data <- ranef(model)$subject %>%
  mutate(subject = rownames(.)) %>%
  mutate(intercept = `(Intercept)`) %>%
  select(c(intercept, subject))
rownames(data) <- NULL

# merge with demo
data <- left_join(data, demo, c("subject" = "participant_id"))

# plot age
p1 <- ggplot(data, aes(x=age, y=intercept, color=sex)) +
  geom_point() +
  geom_hline(aes(yintercept=0), lty="dashed") +
  labs(x="Age", y="Random Intercept")

p2 <- ggplot(data, aes(x=sex, y=intercept, fill=sex)) +
  geom_boxplot(notch=TRUE) +
  geom_hline(aes(yintercept=0), lty="dashed") +
  labs(x="Sex", y="Random Intercept") +
  guides(fill = "none") # remove legend for "fill"

p3 <- ggplot(data, aes(x=intercept, fill=handedness)) +
  geom_histogram() + 
  geom_vline(aes(xintercept=0), lty="dashed") +
  labs(x="Random Intercept", y="Count") +
  scale_fill_viridis_d()
  
ggarrange(p1,p2,p3)









data_long <- data %>%
  pivot_longer(
    cols = rownameΩs(.), #-c("subject"), #, # Select columns starting with "est"
    names_to = "subject",         # Create the "level" column
    values_to = "mean" # Create the "conditional mean" column
  ) 

if (any(startsWith(data_long$level, "experiment"))) {
  title <- "ALL" 
} else {
  title <- unique(orig_data$experiment)
} # TODO: can I get the experiment information from somewhere in the model?
