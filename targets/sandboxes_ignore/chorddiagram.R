
# CHORD diagram to visualize set interactions in Models
# 

library(circlize)
# Create an adjacency matrix: 
# a list of connections between 20 origin nodes, and 5 destination nodes:
numbers <- sample(c(1:1000), 100, replace = T)
data <- matrix( numbers, ncol=5)
rownames(data) <- paste0("orig-", seq(1,20))
colnames(data) <- paste0("dest-", seq(1,5))

# Make the circular plot
chordDiagram(data, transparency = 0.5)



# Own data

# create a 8x8 matrix
varnames <- c("ref","hpf","lpf","emc","mac","base","det","ar")
numbers <- c(0,1,1,0,0,0,0,1,
             1,0,1,1,1,1,1,1,
             1,1,0,1,1,1,1,1,
             0,1,1,0,1,0,0,1,
             0,1,1,1,0,0,0,1,
             0,1,1,0,0,0,1,0,
             0,1,1,0,0,1,0,0,
             1,1,1,1,1,0,0,0)

data <- matrix( numbers, ncol=8)
rownames(data) <- varnames
colnames(data) <- varnames

# greyscale
col_fun = colorRamp2(range(data), c("#ddd8d5", "#4e4b44"), transparency = 0.5)


chordDiagram(data, 
             transparency = 0,
             symmetric = TRUE,
             big.gap = 20,
             small.gap = 5,
             link.visible = data > 0.5,
             grid.col = "black",
             col = col_fun,
             #preAllocateTracks = list(track.height = max(strwidth(unlist(dimnames(data)))))
             )
title(main = "Allowed ")
