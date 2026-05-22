# ============================================================
# LINEAR REGRESSION — Student Performance
# Target: G3 (final grade, 0-20, continuous)
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

# ---------- Simple linear regressions (exploratory) ----------
cat("=== Simple Linear Regressions (Math) ===\n")

d1.fail.lm = lm(G3 ~ failures, data = d1.training)
summary(d1.fail.lm)

d1.absence.lm = lm(G3 ~ absences, data = d1.training)
summary(d1.absence.lm)

d1.study.lm = lm(G3 ~ studytime, data = d1.training)
summary(d1.study.lm)

# ---------- Forward stepwise selection (find best features) ----------
library(leaps)

cat("\n=== Forward Stepwise Selection — Math (excluding G1, G2) ===\n")
mod.fw.d1 = regsubsets(G3 ~ . - G1 - G2, nvmax = 8,
                       data = d1.training, method = "forward")
sm.fw.d1  = summary(mod.fw.d1)
print(sm.fw.d1)
cat("BIC values (pick lowest):\n")
print(round(sm.fw.d1$bic, 2))
cat("Adjusted R2 (pick highest):\n")
print(round(sm.fw.d1$adjr2, 3))

best.n.d1 = which.min(sm.fw.d1$bic)
cat(sprintf("\nBest model size by BIC: %d predictors\n", best.n.d1))

# ---------- Full Multiple Linear Regression ----------
cat("\n=== Multiple Linear Regression — Math ===\n")
d1.mlr = lm(G3 ~ . - G1 - G2, data = d1.training)
summary(d1.mlr)

d1.mlr.pred = predict(d1.mlr, newdata = d1.test)
d1.mlr.rmse = sqrt(mean((d1.test$G3 - d1.mlr.pred)^2))
d1.mlr.r2   = 1 - sum((d1.test$G3 - d1.mlr.pred)^2) /
                    sum((d1.test$G3 - mean(d1.test$G3))^2)
cat(sprintf("Math MLR  ->  RMSE: %.3f  |  R2: %.3f\n", d1.mlr.rmse, d1.mlr.r2))

# Residual plots
par(mfrow = c(2, 2))
plot(d1.mlr, main = "Math MLR Diagnostics")
par(mfrow = c(1, 1))

# ---------- Same for Portuguese ----------
cat("\n=== Forward Stepwise Selection — Portuguese (excluding G1, G2) ===\n")
mod.fw.d2 = regsubsets(G3 ~ . - G1 - G2, nvmax = 8,
                       data = d2.training, method = "forward")
sm.fw.d2  = summary(mod.fw.d2)
print(sm.fw.d2)
cat("BIC values:\n")
print(round(sm.fw.d2$bic, 2))
cat("Adjusted R2:\n")
print(round(sm.fw.d2$adjr2, 3))

cat("\n=== Multiple Linear Regression — Portuguese ===\n")
d2.mlr = lm(G3 ~ . - G1 - G2, data = d2.training)
summary(d2.mlr)

d2.mlr.pred = predict(d2.mlr, newdata = d2.test)
d2.mlr.rmse = sqrt(mean((d2.test$G3 - d2.mlr.pred)^2))
d2.mlr.r2   = 1 - sum((d2.test$G3 - d2.mlr.pred)^2) /
                    sum((d2.test$G3 - mean(d2.test$G3))^2)
cat(sprintf("Portuguese MLR  ->  RMSE: %.3f  |  R2: %.3f\n", d2.mlr.rmse, d2.mlr.r2))

par(mfrow = c(2, 2))
plot(d2.mlr, main = "Portuguese MLR Diagnostics")
par(mfrow = c(1, 1))

# ---------- Summary ----------
cat("\n========================================\n")
cat("       LINEAR REGRESSION SUMMARY\n")
cat("========================================\n")
results = data.frame(
  Dataset = c("Math", "Portuguese"),
  RMSE    = round(c(d1.mlr.rmse, d2.mlr.rmse), 3),
  R2      = round(c(d1.mlr.r2,   d2.mlr.r2), 3)
)
print(results)
