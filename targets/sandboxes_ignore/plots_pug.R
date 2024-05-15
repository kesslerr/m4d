library(ggplot2)
library(dplyr)
library(targets)
library(RColorBrewer)

# function to add noise to iris
blur_iris <- function(iris){
  n_rows <- nrow(iris)
  # Standard deviation for the noise
  noise_sd <- 0.3
  # Generate normally distributed noise
  noise <- matrix(rnorm(n_rows * 4, mean = 0, sd = noise_sd), ncol = 4)
  # Add noise to the first four columns
  iris[, 1:4] <- iris[, 1:4] + noise
  return(iris)
}

#data(iris)
#new_iris <- blur_iris(iris)

#colormap from the numerosity paper
colors_dark <- c("#851e3e", "#537d7d", "#3c1d85") # red, green, purple
colors_light <- c("#f6eaef", "#f2fefe", "#9682c0")
purples = c(colors_light[3], colors_dark[3])

# import N170 sliding data
data <- tar_read(data_sliding) %>% 
  filter(experiment == "N170") %>%
  filter(lpf=="20", hpf=="0.1", emc=="ica", mac=="ica", ref=="average", base=="400ms", det=="linear", ar=="false")

p1 <- ggplot(data, aes(x=times, y=`balanced accuracy`)) +
  geom_line(lwd=2) +
  geom_hline(yintercept = 0.5, linetype="dotted", lwd=2) +
  geom_vline(xintercept = 0., linetype="dotted", lwd=2) +
  theme_minimal() +
  theme(
    text = element_text(size = 20, face = "bold"),
    axis.text = element_text(size = 18, face = "bold"),
    axis.title = element_text(size = 20, face = "bold"),
    axis.line = element_blank(), #element_line(color = "black", size = 2),
    axis.ticks = element_blank(), #element_line(color = "black", size = 2),
    panel.grid.major = element_blank(), #element_line(color = "gray", size = 1),
    panel.grid.minor = element_blank(), #element_line(color = "gray", size = 1)
  ) +
  labs(y="Accuracy", x="Time")
p1

# 2d plane
# 

data(iris)
iris <- blur_iris(iris)
iris <- iris %>% filter(Species %in% c("virginica", "versicolor"))
set.seed(123)

p2a <- ggplot(iris, aes(x=Petal.Width, y=Sepal.Width, z=Petal.Length, color=Species)) + 
  geom_point(size=3) +
  theme_minimal() +
  theme(
    text = element_text(size = 20, face = "bold"),
    axis.text = element_blank(), #element_text(size = 18, face = "bold"),
    axis.title = element_text(size = 20, face = "bold"),
    axis.line = element_blank(), #element_line(color = "black", size = 2),
    axis.ticks = element_blank(), #element_line(color = "black", size = 2),
    panel.grid.major = element_blank(), #element_line(color = "gray", size = 1),
    panel.grid.minor = element_blank(), #element_line(color = "gray", size = 1)
  ) +
  scale_color_manual(values=purples) +
  theme(legend.position = "none") +
  labs(x="Channel 1", y="Channel 2") +
  geom_abline(intercept = -5, slope = 5, color = "black", size = 2)
p2a


data(iris)
iris <- blur_iris(iris)
iris <- iris %>% filter(Species %in% c("virginica", "versicolor"))
# invert the numbers of Petal.Width and Sepal.Width
#iris <- iris %>% mutate(Petal.Width = -Petal.Width, Sepal.Width = -Sepal.Width)
# shuffle 10% of of the "species"
#iris <- iris %>% mutate(Sepal.Width = Sepal.Width + rnorm(nrow(iris), 0, 0.2))

set.seed(123)
p2b <- ggplot(iris, aes(x=Petal.Length, y=Sepal.Width, color=Species)) + 
  geom_point(size=3) +
  theme_minimal() +
  theme(
    text = element_text(size = 20, face = "bold"),
    axis.text = element_blank(), #
    #axis.text = element_text(size = 18, face = "bold"),
    axis.title = element_text(size = 20, face = "bold"),
    axis.line = element_blank(), #element_line(color = "black", size = 2),
    axis.ticks = element_blank(), #element_line(color = "black", size = 2),
    panel.grid.major = element_blank(), #element_line(color = "gray", size = 1),
    panel.grid.minor = element_blank(), #element_line(color = "gray", size = 1)
  ) +
  scale_color_manual(values=purples) +
  theme(legend.position = "none") +
  labs(x="Channel 1", y="") + #, y="Channel 2"
  geom_abline(intercept = -100, slope = 21, color = "black", size = 2)
p2b

data(iris)
iris <- blur_iris(iris)
iris <- iris %>% filter(Species %in% c("virginica", "versicolor"))
# invert the numbers of Petal.Width and Sepal.Width
#iris <- iris %>% mutate(Petal.Width = -Petal.Width, Sepal.Width = -Sepal.Width)
# shuffle 10% of of the "species"
#iris <- iris %>% mutate(Sepal.Width = Sepal.Width + rnorm(nrow(iris), 0, 0.2))

set.seed(123)
p2c <- ggplot(iris, aes(x=Petal.Width, y=Petal.Length, color=Species)) + 
  geom_point(size=3) +
  theme_minimal() +
  theme(
    text = element_text(size = 20, face = "bold"),
    axis.text = element_blank(), #
    #axis.text = element_text(size = 18, face = "bold"),
    axis.title = element_text(size = 20, face = "bold"),
    axis.line = element_blank(), #element_line(color = "black", size = 2),
    axis.ticks = element_blank(), #element_line(color = "black", size = 2),
    panel.grid.major = element_blank(), #element_line(color = "gray", size = 1),
    panel.grid.minor = element_blank(), #element_line(color = "gray", size = 1)
  ) +
  scale_color_manual(values=purples) +
  theme(legend.position = "none") +
  labs(x="Channel 1", y="") +
  geom_abline(intercept = +20, slope = -9, color = "black", size = 2)
p2c

### Legend
data(iris)
iris <- iris %>% filter(Species %in% c("virginica", "versicolor"))
iris$Species <- ifelse(iris$Species == "virginica", "A", 
                       ifelse(iris$Species == "versicolor", "B", iris$Species))
names(iris)[5] <- "Class"

p2leg <- ggplot(iris, aes(x=Petal.Width, y=Petal.Length, color=Class)) + 
  geom_point(size=4) +
  scale_color_manual(values=purples) +
  theme_minimal() +
  theme(
    text = element_text(size = 20, face = "bold"),
    axis.text = element_blank(), #
    #axis.text = element_text(size = 18, face = "bold"),
    axis.title = element_text(size = 20, face = "bold"),
    axis.line = element_blank(), #element_line(color = "black", size = 2),
    axis.ticks = element_blank(), #element_line(color = "black", size = 2),
    panel.grid.major = element_blank(), #element_line(color = "gray", size = 1),
    panel.grid.minor = element_blank(), #element_line(color = "gray", size = 1)
  ) +
  labs(color="Class")
p2leg 
legend <- as_ggplot(ggpubr::get_legend(p2leg))
legend

### ARROWS
arrow_data <- data.frame(x = c(10, 5),  y = c(1, 10))
ar1 <- ggplot() +
  geom_segment(data = arrow_data, aes(x = x[1], y = y[1], xend = x[2], yend = y[2]),
               arrow = arrow(type = "closed", length = unit(0.3, "inches")), size = 1) +
  xlim(0, 11) + ylim(0, 11) +   # Adjust limits to fit the arrow
  theme_void()  # Remove axes and labels
ar1

arrow_data <- data.frame( x = c(5, 5),y = c(1, 10)
)
ar2 <- ggplot() +
  geom_segment(data = arrow_data, aes(x = x[1], y = y[1], xend = x[2], yend = y[2]),
               arrow = arrow(type = "closed", length = unit(0.3, "inches")), size = 1) +
  xlim(0, 11) + ylim(0, 11) +   # Adjust limits to fit the arrow
  theme_void()  # Remove axes and labels
ar2

arrow_data <- data.frame(x = c(1, 5),y = c(1, 10))
ar3 <- ggplot() +
  geom_segment(data = arrow_data, aes(x = x[1], y = y[1], xend = x[2], yend = y[2]),
               arrow = arrow(type = "closed", length = unit(0.3, "inches")), size = 1) +
  xlim(0, 11) + ylim(0, 11) +   # Adjust limits to fit the arrow
  theme_void()  # Remove axes and labels
ar3



# Blank plot placeholder
blank_plot <- ggplot() +
  theme_void()

# collate them to subplots
library(ggpubr)

ggarrange(
  # Second row with box and dot plots
  ggarrange(legend, p2a, p2b, p2c, ncol = 4), 
  ggarrange(blank_plot, ar1, ar2, ar3, ncol = 4),
  p1,                # First row with line plot
  nrow = 3, 
  labels = ""       # Label of the line plot
) 

ggsave("/Users/roman/GitHub/m4d/presentation/plots/timeresolved_schema.png", dpi=300, width=10, height=7)
