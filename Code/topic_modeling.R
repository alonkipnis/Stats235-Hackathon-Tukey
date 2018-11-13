library(readr)
#find topics in congressional records using LDA
library(dplyr)

library(tidyverse)

library(topicmodels)
library(tidytext)
library(tm)

raw.corpus <- read_csv("./CongRec/speech_w_data.csv")

unit1 <- raw.corpus %>%
    filter(congress_id == 114) %>%
    select(speech_id, speech)

#unit2 <- raw.corpus %>%
#    filter(party == 'D', congress_id == 114) %>%
#    select(speech_id, speech)

unit1.dt <- unit1 %>%
    unnest_tokens(word, speech) %>%
    mutate(word = str_extract(word, "[a-z']+")) %>%
    count(speech_id, word, sort = TRUE) %>%
    bind_tf_idf(word, speech_id, n)

    DTM <- unit1.dt %>%
  cast_dtm(speech_id, word, n)

ap_lda <- LDA(DTM, k = 15, control = list(seed = 1234))

save(ap_lda, file = "result.RData")