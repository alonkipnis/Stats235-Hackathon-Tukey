two_unit_test = function(unit1, unit2, words_to_ignore) {
  # unit1, unit2 are dataframes with columns: speech_id (integer) , speech (string)
  # wrods_to_ignore is a list of words to be discarded from the count (such as function/noninteresting word)
  
  #function returns a dataframe containing Z scores of p-values and a list of their associated words
  # HC* ~ max(zz)
  
  library(tidytext)
  library(SnowballC)
  
  unit1.words <- unit1 %>%
    unnest_tokens(word, speech, token = "words") %>%
    mutate(word = str_extract(word, "[a-z']+")) %>%
    drop_na %>% 
    mutate(word = wordStem(word)) %>%
    count(word, sort = TRUE) %>%
    mutate(total = sum(n)) 
  
  unit2.words <- unit2 %>%
    unnest_tokens(word, speech, token = "words") %>%
    mutate(word = str_extract(word, "[a-z']+")) %>%
    drop_na %>% 
    mutate(word = wordStem(word)) %>%
    count(word, sort = TRUE) %>%
    mutate(total = sum(n)) 
  
  word.counts <- unit1.words %>% 
    inner_join(unit2.words, by = 'word') %>%
    filter(!(word %in% c(ignore_list, function_words, additional_words1, additional_words2, singletons))) %>%
    mutate(total = total.x + total.y) %>%
    filter(total > 15) %>%
    rowwise() %>%
    mutate(p = (n.x+n.y) / (total.x + total.y)) %>%
    mutate(se = sqrt(p*(1-p)*(1/total.x + 1/ total.y))) %>%
    mutate(z.score = (n.x /total.x - n.y / total.y) / se) %>%
    mutate(pval = 2*pnorm(-abs(z.score))) %>%
    mutate(pval2 = binom.test(n = nn.x+nn.y, x = nn.x,
                              p = (total.x - nn.x) / (total.y + total.x - nn.x - nn.y),
                              alt = "two.sided")$p.value) %>%. #pvalue based on exact binomial test
  select(word, n.x, n.y, pval, pval2) 
  
  source("./HC_aux.R") #load functions for computing HC
  hc = hc.vals(word.counts$pval2, alpha = 0.4) #alpha determine the lowest fraction of p-values to consider. Usually < 0.5
  #hc.star <- hc$hc.star
  data_frame(uu = hc$uu, zz = hc$z, pp = hc$p.sorted, word = word.counts$word[hc$p.sorted_idx])
} 