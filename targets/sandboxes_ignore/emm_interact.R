# EMM with interactions
# https://cran.r-project.org/web/packages/emmeans/vignettes/interactions.html
# auto noise dataset
library(grid)
library(gridExtra)
library(ggplot2)
library(dplyr)
library(cowplot)
library(ggpubr)
library(pals)  

# toy example
noise.lm <- lm(noise/10 ~ size * type * side, data = auto.noise)
anova(noise.lm)

emmeans(noise.lm, pairwise ~ size)

emmeans(noise.lm, pairwise ~ size | type)

emmip(noise.lm, type ~ size | side)



# my data

model <- tar_read(sliding_LMi2, branches=1)[[1]]
data <- tar_read(data_tsum_exp, branches=1)
#experiment <- "ERN"


#emmip(model, emc ~ lpf, data=data)
#emmip(model, lpf ~ hpf | ref | ar, data=data)



# loop over all combinations
experiment <- experiment <- unique(data$experiment)
variables = c("ref", "hpf","lpf","emc","mac","base","det","ar")

#results = data.frame()
means = data.frame()
contra = data.frame()
for (variable.1 in variables) {
  for (variable.2 in variables) {
    if (variable.1 != variable.2) {
      #print(paste(variable.1, variable.2))
      
      # extract marginal means grouped for results and stats
      emm <- emmeans(model, as.formula(paste("pairwise ~", variable.1, "|", variable.2)), data=data)
      
      # means
      dfw <- emm$emmeans %>% 
        as.data.frame() # leaving out contrasts for now
      dfw$variable.1 <- names(dfw)[1] # grouping variable 
      dfw$variable.2 <- names(dfw)[2] 
      names(dfw)[1] <- "level.1" # grouping variable
      names(dfw)[2] <- "level.2" 
      dfw <- dfw[, c(8, 9, 1, 2, 3)]  # CAVE: the SD/CIs can not be used (see warning and values), therefore cutting them
      # to avoid TRUE and FALSE being converted to NA (in variable="ar")
      if (class(dfw$level.1) == "logical") {dfw$level <- as.factor(dfw$level.1)}
      if (class(dfw$level.2) == "logical") {dfw$level <- as.factor(dfw$level.2)}
      means <- rbind(means, dfw)
      
      # contrasts
      dfc <- emm$contrasts %>% 
        as.data.frame() %>% # leaving out contrasts for now
        mutate(variable.1 = variable.1) %>%
        mutate(variable.2 = variable.2) %>%
        separate(contrast, c("level.1.1", "level.1.2"), " - ")
      names(dfc)[3] <- "level.2"
      dfc <- dfc[, c(9, 10, 1, 2, 3, 4, 5, 6, 7, 8)]
      contra <- rbind(contra, dfc)
      
      # f test # TODO: check if and how to do it
      #f <- joint_tests(emm)
      
      # extract marginal means grouped as plotting information
      #tmp <- emmip(model, as.formula(paste(variable.1, "~", variable.2)), data=data, plotit=FALSE)
      #tmp$variable.1 <- names(tmp)[1]
      #tmp$variable.2 <- names(tmp)[2]
      #names(tmp)[1:2] <- c("level1","level2")
      #results <- rbind(results, tmp)
      #emmip(model, ref ~ hpf, data=data)
    }
  }
}



## plot results

### 1 large line facet plot
meansr <- means %>% 
  mutate(variable.1 = recode(variable.1, !!!replacements)) %>%
  mutate(variable.2 = recode(variable.2, !!!replacements))

cols_stepped <- stepped(20)
cols <- c("None" = "black",
          "0.1" = cols_stepped[1],    
          "0.5" = cols_stepped[9],     
          "6" = cols_stepped[1],       
          "20" = cols_stepped[9],      
          "45" = cols_stepped[17],      
          "ica" = cols_stepped[1],     
          "200ms" = "black",   
          "400ms" = cols_stepped[1],   
          "offset" = "black",  
          "linear" = cols_stepped[1], 
          "false" = "black",
          "true" = cols_stepped[1],
          "average" = "black",
          "Cz" = cols_stepped[1],
          "P9P10" = cols_stepped[9]
          )

p1 <- ggplot(meansr, 
       aes(x = level.1, y = emmean, col = level.2, group = level.2)) + 
  geom_line(size = 1.2) + 
  facet_grid(variable.2~variable.1, scales = "free") +
  labs(title = experiment, y = "Marginal Mean", x = "Level of Model Term 1", color = "Level of\nModel Term 2") +
  # TODO: better y value titel
  scale_color_manual(values=cols) +
  theme_classic() +
  scale_x_discrete(expand = c(0.2, 0.0)) + # strech a bit in x direction
  theme(legend.position = "none")  # Remove legend
  #scale_color_manual(values=as.vector(stepped(20)))
  
p1

# manually make a colorframe
variable.2s <- sort(unique(meansr$variable.2))

# debug
#v2 <- variable.2s[1]
legends <- list()
for (v2 in variable.2s){
  results_filtered <- meansr %>% filter(variable.2 == v2)
  ptmp <- ggplot(results_filtered, 
               aes(x = level.1, y = emmean, col = level.2, group = level.2)) + 
    geom_line(size = 1.2) + 
    facet_grid(.~variable.1, scales = "free") +
    labs(color = v2) +
    scale_color_manual(values=cols) +
    theme_classic()
  
  # get legend
  legend <- as_ggplot(ggpubr::get_legend(ptmp))
  legends <- c(legends, list(legend))
}


# possibility 1: legend at the right side
# p1_and_legends <- c(list(p1), legends)
# grid.arrange(grobs=p1_and_legends, #, ncol = 5)y
#           layout_matrix = matrix(c(1,1,1,1,1,1,1,2,
#                                    1,1,1,1,1,1,1,3,
#                                    1,1,1,1,1,1,1,4,
#                                    1,1,1,1,1,1,1,5,
#                                    1,1,1,1,1,1,1,6,
#                                    1,1,1,1,1,1,1,7,
#                                    1,1,1,1,1,1,1,8,
#                                    1,1,1,1,1,1,1,9),
#                                  nrow=8, byrow=TRUE))

# possibility 2: on diagonals
cow <- cowplot::ggdraw() + 
  cowplot::draw_plot(p1, x = 0, y = 0, width = 1.0, height = 1.0) 

d <- 1/8.5
e <- 1/9
for (i in 0:7){
  # single legends
  cow <- cow + cowplot::draw_plot(legends[[i+1]], 
                                  x = d*i+0.05, 
                                  y = 0.875-e*i, 
                                  width = 0.1, height = 0.03)
  # horizontal lines between facets
  if (i<7){
    cow <- cow + cowplot::draw_line(x = c(0.05, 0.97), 
                                    y = c(0.83-e*i, 0.83-e*i), 
                                    color = "grey", size = 0.5)
  }
}
cow

####
