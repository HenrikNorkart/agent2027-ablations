# AGENT2027 final read-out (2026-11-02): GLMMs of record for the contrasts prepared by final_tests.py prep.
# Usage (container ro27-rstats:4.4.1): Rscript final_glmm.R <dir>. Writes <dir>/glmm_results.json.
suppressMessages({library(lme4); library(logistf); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE); d <- args[1]
specs <- fromJSON(file.path(d, "glmm_specs.json"), simplifyVector = FALSE)
out <- list()
for (s in specs) {
  df <- read.csv(file.path(d, s$file), stringsAsFactors = FALSE)
  r <- list(id = s$id, formula = s$formula, n = nrow(df))
  if (nrow(df) == 0 || length(unique(df$y)) < 2) { r$converged <- FALSE; r$note <- "no variation in outcome"; out[[s$id]] <- r; next }
  if (!is.null(s$firth) && isTRUE(s$firth)) {
    f <- tryCatch(logistf(as.formula(s$formula), data = df), error = function(e) e)
    if (inherits(f, "error")) { r$converged <- FALSE; r$note <- conditionMessage(f) } else {
      i <- which(names(coef(f)) == s$term)
      r$estimate <- unname(coef(f)[i]); r$se <- sqrt(diag(vcov(f)))[i]; r$p_wald <- unname(f$prob[i])
      r$ci95 <- c(f$ci.lower[i], f$ci.upper[i]); r$converged <- TRUE }
    out[[s$id]] <- r; next }
  ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
  m <- tryCatch(withCallingHandlers(glmer(as.formula(s$formula), data = df, family = binomial, control = ctrl),
                warning = function(w) { r$warnings <<- c(r$warnings, conditionMessage(w)); invokeRestart("muffleWarning") }),
                error = function(e) e)
  if (inherits(m, "error")) { r$converged <- FALSE; r$note <- conditionMessage(m); out[[s$id]] <- r; next }
  cf <- summary(m)$coefficients
  term <- if (s$term %in% rownames(cf)) s$term else grep(gsub(":", ".*", s$term), rownames(cf), value = TRUE)[1]
  r$estimate <- cf[term, 1]; r$se <- cf[term, 2]; r$p_wald <- cf[term, 4]
  r$ci95 <- c(r$estimate - 1.959964 * r$se, r$estimate + 1.959964 * r$se)
  msgs <- m@optinfo$conv$lme4$messages
  r$converged <- is.null(msgs) && is.null(r$warnings); r$conv_messages <- msgs
  r$singular_all <- isSingular(m) && all(sapply(VarCorr(m), function(v) v[1]) < 1e-6)
  r$singular <- isSingular(m); r$varcorr <- lapply(VarCorr(m), function(v) v[1])
  if (isTRUE(s$lrt)) {
    red <- update(m, as.formula(paste(". ~ . -", s$term)))
    a <- anova(red, m); r$p_lrt <- a[["Pr(>Chisq)"]][2]; r$lrt_chisq <- a[["Chisq"]][2]
    r$or <- exp(r$estimate)
    pr <- tryCatch(confint(m, parm = term, method = "profile", quiet = TRUE), error = function(e) NULL)
    if (!is.null(pr)) r$or_ci95_profile <- exp(as.numeric(pr))
  }
  out[[s$id]] <- r
}
write(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), file.path(d, "glmm_results.json"))
cat("glmm: wrote", length(out), "fits\n")
