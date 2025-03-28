# save the current R environment to file

# navigate to the R env
setwd("../env")

Rversion <- R.version.string
# save version string to txt file
write(Rversion, "Rversion.txt")

# make sure to have sourced the activate.R file before running

# save the installed packages to a csv file
installed_packages <- as.data.frame(installed.packages()[, c("Package", "Version")])
write.csv(installed_packages, "R_packages.csv", row.names = FALSE)
