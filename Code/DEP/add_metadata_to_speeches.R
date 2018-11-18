#read speaches and speach info data and merge to a single table.
library(readr)
library(dplyr)


MIN_COUNT = 300 #only consider speeches with more than MIN_COUNT word

raw.corpus = data.frame()

path_to_data = "./hein-daily"

for(i in 97:114) {  #congress range (97-114)
    speeches <- read_delim(sprintf("%s/speeches_%03d.txt",path_to_data,i), delim = "|",
                           col_types = cols(speech_id = col_integer(), speech = 'c'), quote = '')
    info_party <- read_delim(sprintf("%s/%03d_SpeakerMap.txt",path_to_data,i),  delim = "|") %>%
                    select(speech_id, chamber, party)
    info_date_wordcount <- read_delim(sprintf("%s/descr_%03d.txt",path_to_data,i), delim = "|") %>%
                            select(speech_id, date, word_count)
    speeches_with_data <- speeches %>%
        mutate(congress_id = i) %>%
        inner_join(info_party, by = 'speech_id') %>%
        inner_join(info_date_wordcount, by = 'speech_id') %>% 
        filter(word_count > MIN_COUNT) %>%
        select(speech_id, date, congress_id, chamber, party, speech)
    raw.corpus <- rbind(raw.corpus, speeches_with_data)
}

write_csv(raw.corpus, path = "speech_w_data.csv")
