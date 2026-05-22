# ============================================================
# LOGISTIC REGRESSION — Student Performance
# Target: pass/fail (G3 >= 10 = pass, G3 < 10 = fail)
# G1 and G2 excluded — they are mid-term grades, not pre-term predictors
# ============================================================

# ---------- Load data ----------
d1 = read.table("student-mat.csv", sep = ";", header = TRUE)
d2 = read.table("student-por.csv", sep = ";", header = TRUE)

# ---------- Encode categorical variables ----------
encode = function(df) {
  df$school     = ifelse(df$school == "GP", 0, 1)
  df$sex        = ifelse(df$sex == "F", 0, 1)
  df$address    = ifelse(df$address == "R", 0, 1)
  df$famsize    = ifelse(df$famsize == "LE3", 0, 1)
  df$Pstatus    = ifelse(df$Pstatus == "A", 0, 1)
  df$paid       = ifelse(df$paid == "no", 0, 1)
  df$activities = ifelse(df$activities == "no", 0, 1)
  df$nursery    = ifelse(df$nursery == "no", 0, 1)
  df$higher     = ifelse(df$higher == "no", 0, 1)
  df$internet   = ifelse(df$internet == "no", 0, 1)
  df$romantic   = ifelse(df$romantic == "no", 0, 1)
  df$Mjob       = factor(df$Mjob)
  df$Fjob       = factor(df$Fjob)
  df$reason     = factor(df$reason)
  df$guardian   = factor(df$guardian)
  df$schoolsup  = factor(df$schoolsup)
  df$famsup     = factor(df$famsup)
  df
}

d1 = encode(d1)
d2 = encode(d2)

# Create binary pass/fail target: 10/20 is the minimum pass mark
d1$pass = ifelse(d1$G3 >= 10, 1, 0)
d2$pass = ifelse(d2$G3 >= 10, 1, 0)

cat(sprintf("Math      — Pass: %d  Fail: %d  (%.1f%% pass rate)\n",
            sum(d1$pass), sum(d1$pass == 0), 100 * mean(d1$pass)))
cat(sprintf("Portuguese — Pass: %d  Fail: %d  (%.1f%% pass rate)\n",
            sum(d2$pass), sum(d2$pass == 0), 100 * mean(d2$pass)))

# ---------- Train / Test split ----------
set.seed(42)

n1 = nrow(d1)
idx1        = sample(1:n1, size = 0.8 * n1)
d1.training = d1[idx1, ]
d1.test     = d1[-idx1, ]

n2 = nrow(d2)
idx2        = sample(1:n2, size = 0.8 * n2)
d2.training = d2[idx2, ]
d2.test     = d2[-idx2, ]

# ---------- Helper: evaluation metrics ----------
eval_logit = function(actual, prob, threshold = 0.5) {
  pred = ifelse(prob > threshold, 1, 0)
  cm   = table(Actual = actual, Predicted = pred)
  print(cm)
  acc  = mean(pred == actual)
  tp   = sum(pred == 1 & actual == 1)
  fp   = sum(pred == 1 & actual == 0)
  fn   = sum(pred == 0 & actual == 1)
  prec = ifelse((tp + fp) == 0, NA, tp / (tp + fp))
  rec  = ifelse((tp + fn) == 0, NA, tp / (tp + fn))
  cat(sprintf("Accuracy:  %.3f\n", acc))
  cat(sprintf("Precision: %.3f\n", prec))
  cat(sprintf("Recall:    %.3f\n", rec))
  invisible(list(accuracy = acc, precision = prec, recall = rec))
}

# ---------- Logistic Regression — Math ----------
cat("\n=== Logistic Regression — Math ===\n")
d1.logit = glm(pass ~ . - G1 - G2 - G3, data = d1.training, family = binomial)
summary(d1.logit)

# Odds ratios: exp(coef) shows how each unit increase changes the odds of passing
cat("\nOdds Ratios (exp of coefficients):\n")
print(round(exp(coef(d1.logit)), 3))

d1.logit.prob = predict(d1.logit, newdata = d1.test, type = "response")
cat("\nConfusion Matrix (threshold = 0.5):\n")
d1.metrics = eval_logit(d1.test$pass, d1.logit.prob)

# Predicted probability distribution
hist(d1.logit.prob,
     main = "Math: Predicted Pass Probability",
     xlab = "P(pass)", col = "steelblue", breaks = 20)
abline(v = 0.5, col = "red", lty = 2)

# ---------- Logistic Regression — Portuguese ----------
cat("\n=== Logistic Regression — Portuguese ===\n")
d2.logit = glm(pass ~ . - G1 - G2 - G3, data = d2.training, family = binomial)
summary(d2.logit)

cat("\nOdds Ratios:\n")
print(round(exp(coef(d2.logit)), 3))

d2.logit.prob = predict(d2.logit, newdata = d2.test, type = "response")
cat("\nConfusion Matrix (threshold = 0.5):\n")
d2.metrics = eval_logit(d2.test$pass, d2.logit.prob)

hist(d2.logit.prob,
     main = "Portuguese: Predicted Pass Probability",
     xlab = "P(pass)", col = "darkorange", breaks = 20)
abline(v = 0.5, col = "red", lty = 2)

# ---------- Summary ----------
cat("\n========================================\n")
cat("     LOGISTIC REGRESSION SUMMARY\n")
cat("========================================\n")
results = data.frame(
  Dataset   = c("Math", "Portuguese"),
  Accuracy  = round(c(d1.metrics$accuracy,  d2.metrics$accuracy), 3),
  Precision = round(c(d1.metrics$precision, d2.metrics$precision), 3),
  Recall    = round(c(d1.metrics$recall,    d2.metrics$recall), 3)
)
print(results)
