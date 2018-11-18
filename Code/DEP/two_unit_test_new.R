two_unit_test_new = function(unit1, unit2, list_of_words) {
  # unit1, unit2 are dataframes with columns: speech_id (integer) , speech (string)
  # list_of_words is a list of words to be kept from the count 
  
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
    mutate(total = total.x + total.y) %>%
    filter(word %in% list_of_words) %>%
    filter(n.x + n.y > 15) %>%
	rowwise() %>%
    mutate(p = (n.x+n.y) / (total.x + total.y)) %>%
    mutate(se = sqrt(p*(1-p)*(1/total.x + 1/ total.y))) %>%
    mutate(z.score = (n.x /total.x - n.y / total.y) / se) %>%
    mutate(pval = 2*pnorm(-abs(z.score))) %>%
    mutate(pval2 = binom.test(n = n.x+n.y, x = n.x,
                              p = (total.x - n.x) / (total.y + total.x - n.x - n.y),
                              alt = "two.sided")$p.value) %>% #pvalue based on exact binomial test
  select(word, n.x, n.y, pval, pval2) 
  
  source("./HC_aux.R") #load functions for computing HC
  hc = hc.vals(word.counts$pval2, alpha = 0.25) #alpha determine the lowest fraction of p-values to consider. Usually < 0.5
  #hc.star <- hc$hc.star
  data_frame(uu = hc$uu, zz = hc$z, pp = hc$p.sorted, word = word.counts$word[hc$p.sorted_idx])
} 