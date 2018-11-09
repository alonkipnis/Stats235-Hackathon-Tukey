
p.values.binom_half = function(k1, k2, min.counts = 3, alt = "two.sided"){
  # returns a list of p values from two sample counts k1 and k2
      pv = function(n1, n2, p.null){
            N = n1 + n2;
            p.value = NA;
            if(N >= min.counts){ #at least min.counts
                     test = binom.test(x = n1, n = N, p = p.null, alternative = alt);
                     p.value = test$p.value;
                    }
            p.value
      }
  p = rep(0, length(k1));
  sum.1 = sum(k1)
  sum.2 = sum(k2)
  for( i in 1:length(p) ){
        #p.null = (sum.1 - k1[i]) / (sum.1 + sum.2 - k1[i] - k2[i])  #refined null prob (usually very close to 0.5)
        p.null = 0.5
        p[i] = pv(k1[i], k2[i], p.null);
  }
  p
}

p.values.normal = function(x, alt = "two.sided") {
  p = rep(NA, length(x));
  p = 2*pnorm(-abs(z));
  p
}

z.proportion.test = function(k1, k2) {
n1 = sum(k1)
n2 = sum(k2)
n = k1 + k2
p = (k1 + k2) / (n1 + n2) #pooled prob of success
se = sqrt(p*(1-p)*(1/n1 + 1/n2)) + 1e-10
z =  (k1/n1 - k2/n2) / se 
pv.z = 2*pnorm(-abs(z)) 
pv.z
}


p.values.poisson = function(x, N, r, min.counts = 0, alt = "two.sided"){
  # returns a list of p-values from a poission test
  # N time, #x no. of events #r = rate
  p = rep(NA, length(x));
  for( i in 1:length(p) ){
    if(x[i] >= min.counts){ #at least min.counts
      test = poisson.test(x = x[i], T = N, r = r[i], alternative = alt);
      p[i] = test$p.value;
    }
  }
  p
}


p.value.doubles = function(k1, k2, alt = "two.sided"){
  #exact test for words whose total count = 2
    all.2 = k1 + k2 == 2;  #mark how many times the sum of counts is 2 
    n.2 = sum(all.2); #sum of these times
    if( n.2 > 0){ 
      n.1 = sum(k1[all.2] == 1); #number of single counts
      test=binom.test(x = n.1, n = n.2, p = 0.5, alternative = "two.sided");
      q=test$p.value;
    }}

p.value.triples = function(k1, k2, alt = "two.sided") {
  #exact test for words whose total count = 3
  n30 = sum((k1 == 3) * (k2 == 0))
  n12 = sum((k1 == 1) * (k2 == 2))
  n21 = sum((k1 == 2) * (k2 == 1))
  n03 = sum((k1 == 0) * (k2 == 3))
  all.3 = n30 + n12 + n21 + n03
  if(all.3 > 0) {
    test=binom.test(x = n30 + n12, n = all.3, p = 0.5, alternative = "two.sided");
    q=test$p.value
  }
}

p.value.ncount = function(k1, k2, n, alt = "two.sided"){
  #p-value of distribution of counts equals n  
    n1 = sum(k1==n)
    n2 = sum(k2==n)
    if( n1 + n2 > 0){ 
      test=binom.test(n = n1 + n2, x = n1, p = 0.5, alternative = "two.sided");
      q=test$p.value;
    }}


hc.vals = function(pv, alpha = 0.45, interp = FALSE){ 
    # evaluate HC stat and related quantities

    # pv is is a list of p values
    # alpha is fraction of p values to consider
    # if interp == TRUE then interpolation of p-values around 0.5 is performed. Overrides truncation
    
    pv = pv[!is.na(pv)]
    n = length(pv);
    uu = ((1:n) - 0.5) / n; #approximate expectation of p-values 
    srtd = sort(pv, index.return = TRUE);  #sorted p-vals
    ps = srtd$x
    ps_idx = srtd$ix
    p.half = which(abs(ps - 0.5)<0.05) #p-values that are too close to 0.5
    if (interp && length(p.half) > 1) {
      i1 = max(1,p.half[1]-1)
      i2 = min(tail(p.half,1)+1, length(ps))
      sq = seq(from = ps[i1], to = ps[i2], length.out = length(p.half)+2)
      ps[p.half] <- sq[2:(length(sq)-1)]
    }
    #z = (uu - ps) / sqrt(ps * (1 - ps) + 0.01 ) * sqrt(n); #zeroth order HC approach (can be extended) 
    z = (uu - ps) / sqrt( uu * (1 - uu)) * sqrt(n); 

    max.i = floor(alpha * n + 0.5)
    i.max = which.max(z[1:max.i]);
    z.max = z[i.max];
    
    if(i.max == 1){ #if optimal is at the first entry
       i.max.star = 1 + which.max(z[2:max.i]);
       hc.star    = z[i.max.star];
    } 
    else {
       i.max.star = i.max
       hc.star    = z.max
    }
    list(hc = z.max, hc.star = hc.star, i.max = i.max, i.max.star = i.max.star,
        z = z, uu = uu, p = pv, p.sorted = ps, p.sorted_idx = ps_idx);
}

hc.null.p = function(p, N, n.monte = 100, alt = "two.sided") { #p is the vector of probability of each category
    # evalaute HC n.Monte times over two indepndent samples from p 
    # 'p[i]' is i-th probability of success 
    # 'N' is size of sample
    # 'n.monte' is number of repetitions
    # returns a vector of samples of HC  

     generate.counts = function(N,p){ rbinom(length(p), N, p) }
         #samples from binomial dist. with p = p[i],

    hc.null = rep(0,n.monte)
    for( m in 1:n.monte){
        pv = p.values.binom_half(generate.counts(N,p), generate.counts(N,p), alt = alt); #p-vals under null
        l.hc = hc.vals(pv, alpha = 0.15)
        hc.star = l.hc$hc.star;
        hc.null[m] = hc.star;
        }
    hc.null
    }