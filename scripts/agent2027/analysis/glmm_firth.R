suppressMessages(library(logistf))
d <- read.csv("/w/a12_runs.csv")
# expand to fold-episodes (k successes, n-k failures) for the logistic fit
e <- d[rep(seq_len(nrow(d)), d$n), ]
e$y <- unlist(lapply(seq_len(nrow(d)), function(i) c(rep(1, d$k[i]), rep(0, d$n[i] - d$k[i]))))
m <- logistf(y ~ P + A + J + M + PA + PJ + PM + factor(task), data = e)
out <- data.frame(beta = coef(m), se = sqrt(diag(vcov(m))), p_pl = m$prob)
print(round(out[c("P", "A", "J", "M", "PA", "PJ", "PM"), ], 4))
