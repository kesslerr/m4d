# pubmed searches
# 

#install.packages("pubmedR")
#install.packages("bibliometrix")

library(pubmedR)
library(bibliometrix)

api_key = NULL # without API key, 3 requests per second are allowed

query <- "EEG|Electroencephalography*[Title/Abstract] AND english[LA] AND Journal Article[PT] AND 2004:2023[DP]"
query <- "MEG|Magnetoencephalography*[Title/Abstract] AND english[LA] AND Journal Article[PT] AND 2004:2023[DP]"
query <- "fMRI*[Title/Abstract] AND english[LA] AND Journal Article[PT] AND 2004:2023[DP]"

res <- pmQueryTotalCount(query = query, api_key = api_key)
res$total_count