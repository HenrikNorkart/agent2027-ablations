suppressMessages(library(glmmTMB))
d <- read.csv("/w/a12_runs.csv")
f <- cbind(k, n - k) ~ P + A + J + M + PA + PJ + PM + (1 | task)
for (fam in c("betabinomial", "binomial")) {
  m <- glmmTMB(f, data = d, family = fam)
  s <- summary(m)$coefficients$cond
  cat("==", fam, " logLik", as.numeric(logLik(m)), " sd_task", sqrt(as.numeric(VarCorr(m)$cond$task)),
      if (fam == "betabinomial") paste(" phi", sigma(m)) else "", "\n")
  print(round(s[, c(1, 2, 4)], 4))
}
