# ============================================================
# DECISION TREES & RANDOM FORESTS — Student Performance
# Regression variant: predicts G3 (0-20 continuous)
# Classification variant: predicts pass/fail (G3 >= 10)
# G1 and G2 excluded — they are mid-term grades, not pre-term predictors
# ============================================================

library(rpart)
library(rpart.plot)
library(randomForest)

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

# Create pass/fail target for classification models
d1$pass = as.factor(ifelse(d1$G3 >= 10, 1, 0))
d2$pass = as.factor(ifelse(d2$G3 >= 10, 1, 0))

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

# ---------- Helper: regression metrics ----------
reg_metrics = function(actual, pred) {
  rmse = sqrt(mean((actual - pred)^2))
  r2   = 1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2)
  cat(sprintf("RMSE: %.3f  |  R2: %.3f\n", rmse, r2))
  invisible(list(rmse = rmse, r2 = r2))
}

# ---------- Helper: classification metrics ----------
cls_metrics = function(actual, pred) {
  cm  = table(Actual = actual, Predicted = pred)
  print(cm)
  acc = mean(pred == actual)
  cat(sprintf("Accuracy: %.3f\n", acc))
  invisible(list(accuracy = acc))
}

# ==========================================================
# SECTION 1: DECISION TREES
# ==========================================================

cat("\n========================================\n")
cat("         DECISION TREES — MATH\n")
cat("========================================\n")

# Regression tree (predict G3)
cat("\n--- Regression Tree (predicting G3) ---\n")
d1.tree.reg = rpart(G3 ~ . - G1 - G2 - pass, data = d1.training, method = "anova")
rpart.plot(d1.tree.reg, main = "Math: Regression Tree (G3)")
printcp(d1.tree.reg)

d1.tree.reg.pred = predict(d1.tree.reg, newdata = d1.test)
d1.tree.reg.m    = reg_metrics(d1.test$G3, d1.tree.reg.pred)

# Classification tree (predict pass/fail)
cat("\n--- Classification Tree (predicting Pass/Fail) ---\n")
d1.tree.cls = rpart(pass ~ . - G1 - G2 - G3, data = d1.training, method = "class")
rpart.plot(d1.tree.cls, main = "Math: Classification Tree (Pass/Fail)")
printcp(d1.tree.cls)

d1.tree.cls.pred = predict(d1.tree.cls, newdata = d1.test, type = "class")
d1.tree.cls.m    = cls_metrics(d1.test$pass, d1.tree.cls.pred)

cat("\n========================================\n")
cat("       DECISION TREES — PORTUGUESE\n")
cat("========================================\n")

cat("\n--- Regression Tree (predicting G3) ---\n")
d2.tree.reg = rpart(G3 ~ . - G1 - G2 - pass, data = d2.training, method = "anova")
rpart.plot(d2.tree.reg, main = "Portuguese: Regression Tree (G3)")
printcp(d2.tree.reg)

d2.tree.reg.pred = predict(d2.tree.reg, newdata = d2.test)
d2.tree.reg.m    = reg_metrics(d2.test$G3, d2.tree.reg.pred)

cat("\n--- Classification Tree (predicting Pass/Fail) ---\n")
d2.tree.cls = rpart(pass ~ . - G1 - G2 - G3, data = d2.training, method = "class")
rpart.plot(d2.tree.cls, main = "Portuguese: Classification Tree (Pass/Fail)")
printcp(d2.tree.cls)

d2.tree.cls.pred = predict(d2.tree.cls, newdata = d2.test, type = "class")
d2.tree.cls.m    = cls_metrics(d2.test$pass, d2.tree.cls.pred)

# ==========================================================
# SECTION 2: RANDOM FORESTS
# ==========================================================

set.seed(42)

cat("\n========================================\n")
cat("         RANDOM FOREST — MATH\n")
cat("========================================\n")

# Regression forest (predict G3)
cat("\n--- Regression Forest (predicting G3) ---\n")
d1.rf.reg = randomForest(G3 ~ . - G1 - G2 - pass,
                         data = d1.training, ntree = 500, importance = TRUE)
print(d1.rf.reg)

d1.rf.reg.pred = predict(d1.rf.reg, newdata = d1.test)
d1.rf.reg.m    = reg_metrics(d1.test$G3, d1.rf.reg.pred)

varImpPlot(d1.rf.reg, main = "Math RF (Regression): Variable Importance")

# Classification forest (predict pass/fail)
cat("\n--- Classification Forest (predicting Pass/Fail) ---\n")
d1.rf.cls = randomForest(pass ~ . - G1 - G2 - G3,
                         data = d1.training, ntree = 500, importance = TRUE)
print(d1.rf.cls)

d1.rf.cls.pred = predict(d1.rf.cls, newdata = d1.test)
d1.rf.cls.m    = cls_metrics(d1.test$pass, d1.rf.cls.pred)

varImpPlot(d1.rf.cls, main = "Math RF (Classification): Variable Importance")

cat("\n========================================\n")
cat("       RANDOM FOREST — PORTUGUESE\n")
cat("========================================\n")

cat("\n--- Regression Forest (predicting G3) ---\n")
d2.rf.reg = randomForest(G3 ~ . - G1 - G2 - pass,
                         data = d2.training, ntree = 500, importance = TRUE)
print(d2.rf.reg)

d2.rf.reg.pred = predict(d2.rf.reg, newdata = d2.test)
d2.rf.reg.m    = reg_metrics(d2.test$G3, d2.rf.reg.pred)

varImpPlot(d2.rf.reg, main = "Portuguese RF (Regression): Variable Importance")

cat("\n--- Classification Forest (predicting Pass/Fail) ---\n")
d2.rf.cls = randomForest(pass ~ . - G1 - G2 - G3,
                         data = d2.training, ntree = 500, importance = TRUE)
print(d2.rf.cls)

d2.rf.cls.pred = predict(d2.rf.cls, newdata = d2.test)
d2.rf.cls.m    = cls_metrics(d2.test$pass, d2.rf.cls.pred)

varImpPlot(d2.rf.cls, main = "Portuguese RF (Classification): Variable Importance")

# ==========================================================
# SECTION 3: MODEL COMPARISON SUMMARY
# ==========================================================

cat("\n========================================\n")
cat("           MODEL COMPARISON\n")
cat("========================================\n")

cat("\n--- Regression (predicting G3, lower RMSE / higher R2 is better) ---\n")
reg.results = data.frame(
  Dataset = c(rep("Math", 2), rep("Portuguese", 2)),
  Model   = rep(c("Decision Tree", "Random Forest"), 2),
  RMSE    = round(c(d1.tree.reg.m$rmse, d1.rf.reg.m$rmse,
                    d2.tree.reg.m$rmse, d2.rf.reg.m$rmse), 3),
  R2      = round(c(d1.tree.reg.m$r2,   d1.rf.reg.m$r2,
                    d2.tree.reg.m$r2,   d2.rf.reg.m$r2), 3)
)
print(reg.results)

cat("\n--- Classification (predicting Pass/Fail, higher accuracy is better) ---\n")
cls.results = data.frame(
  Dataset  = c(rep("Math", 2), rep("Portuguese", 2)),
  Model    = rep(c("Decision Tree", "Random Forest"), 2),
  Accuracy = round(c(d1.tree.cls.m$accuracy, d1.rf.cls.m$accuracy,
                     d2.tree.cls.m$accuracy, d2.rf.cls.m$accuracy), 3)
)
print(cls.results)
