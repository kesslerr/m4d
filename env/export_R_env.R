# save the current R environment to file

R.version.string

installed_packages <- as.data.frame(installed.packages()[, c("Package", "Version")])
write.csv(installed_packages, "installed_packages.csv", row.names = FALSE)
