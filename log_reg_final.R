
#  Predicting Student Pass/Fail with LASSO Logistic Regression
#  Datasets : UCI Student Performance (Math + Portuguese)
#  Target   : pass = 1 if G3 >= 10, else 0

#  Domain problem  -> Schools want to flag students at risk of failing
#                     the final exam early enough to intervene.
#  Data sci problem-> Binary classification of pass/fail from
#                     demographic, family and behavioural features.


# Packages
library(glmnet)     # LASSO logistic regression with CV
library(pROC)       # ROC curve and AUC
library(corrplot)   # To plot correlation heatmap

# Setting working directory to wherever the CSVs live
setwd("C:/Users/Ashutosh/Stat5009_group17")

# Load both datasets 
d1 <- read.csv("student-mat.csv", header = T, sep = ";")   # Math
d2 <- read.csv("student-por.csv", header = T, sep = ";")   # Portuguese



# PART 1 : MATH (d1)          

#1. Inspect the datasets
head(d1)
str(d1)
dim(d1)

# 2. Build binary response
# Pass = 1 if final grade G3 >= 10 (Portuguese 0-19 scale)
d1$pass <- ifelse(d1$G3 >= 10, 1, 0)

# Drop G3 - as it would leak the answer into the features
d1 <- d1[ , !names(d1) %in% "G3"]

# To show Class balance
table(d1$pass) / nrow(d1)


# 3. Exploratory Data Analysis

# 3a. Class balance bar chart
par(mfrow = c(1, 1))
barplot(table(d1$pass),
        names.arg = c("Fail", "Pass"),
        col  = c("tomato", "steelblue"),
        main = "Math - Class balance",
        ylab = "Number of students")

# 3b. Boxplots - continuous features vs pass
par(mfrow = c(2, 2))
boxplot(G1 ~ pass, data = d1,
        main = "G1 by Pass (Math)",
        col = c("tomato", "steelblue"))
boxplot(G2 ~ pass, data = d1,
        main = "G2 by Pass (Math)",
        col = c("tomato", "steelblue"))
boxplot(absences ~ pass, data = d1,
        main = "Absences by Pass (Math)",
        col = c("tomato", "steelblue"))
boxplot(studytime ~ pass, data = d1,
        main = "Study time by Pass (Math)",
        col = c("tomato", "steelblue"))

# 3c. Categorical features vs pass rate
par(mfrow = c(2, 2))
spineplot(as.factor(d1$pass) ~ as.factor(d1$sex),
          main = "Pass rate by sex (Math)",
          col = c("tomato", "steelblue"))
spineplot(as.factor(d1$pass) ~ as.factor(d1$address),
          main = "Pass rate by address (Math)",
          col = c("tomato", "steelblue"))
spineplot(as.factor(d1$pass) ~ as.factor(d1$higher),
          main = "Pass rate by wants higher ed (Math)",
          col = c("tomato", "steelblue"))
spineplot(as.factor(d1$pass) ~ as.factor(d1$schoolsup),
          main = "Pass rate by school support (Math)",
          col = c("tomato", "steelblue"))

# 3d. Past failures vs pass rate
par(mfrow = c(1, 1))
barplot(prop.table(table(d1$failures, d1$pass), margin = 1),
        beside = TRUE,
        col = c("tomato", "steelblue"),
        legend.text = c("Fail", "Pass"),
        main = "Past failures vs current pass rate (Math)",
        xlab = "Number of past failures",
        ylab = "Proportion")

# 3e. Correlation heatmap of numeric features
# Shows multicollinearity (G1 and G2 will be highly correlated)
num_cols <- sapply(d1, is.numeric)
cor_mat  <- cor(d1[, num_cols])
corrplot(cor_mat, method = "color", type = "upper",
         tl.cex = 0.7, tl.col = "black",
         title = "Math - correlation of numeric features",
         mar = c(0, 0, 2, 0))
cor(d1.tr$G1, d1.tr$G2)


# 4. Train / Test split 
d1$pass <- as.factor(d1$pass)   # glmnet needs a factor for binomial

n <- nrow(d1)
set.seed(234)
s.in <- sample(1:n, size = 0.8 * n)

d1.tr <- d1[s.in, ]
d1.ts <- d1[-s.in, ]

# Build numeric feature matrix (glmnet needs a matrix, not a data frame)
# model.matrix one-hot encodes categorical variables.
# "-1" drops the intercept column (glmnet adds its own).
x.tr <- model.matrix(pass ~ ., data = d1.tr)[ , -1]
x.ts <- model.matrix(pass ~ ., data = d1.ts)[ , -1]
y.tr <- d1.tr$pass

dim(x.tr)


# 5. Fit LASSO logistic regression 
# alpha = 1 -> LASSO (absolute-value penalty, zeroes out weak features)
# CV picks the best shrinkage parameter (lambda) automatically.
# We chose LASSO Logistic Regression as the model has too many features, 
# which are likely correlated or irrelevant. 
# Two key assumptions of LASSO model are:
# •	A linear relationship exists between the features and the log-odds of the response.
# •	LASSO assumes that the underlying true model is sparse, 
#   meaning only a small subset of the total features have a significant impact on the target variable. 
#   Therefore, it penalizes the absolute values of less important, correlated features to zero. It is therefore known as a feature selection model.
# •	The feature sets are independant (no multicollinearity) is also a key assumption


set.seed(234)
cv.l <- cv.glmnet(x = x.tr, y = y.tr,
                  alpha = 1,
                  family = "binomial",
                  nfolds = 10,
                  type.measure = "deviance")
plot(cv.l)
title("Math - LASSO CV", line = 2.5)

# G1 and G2 are highly correlated but dropping G2 increases CV -error
x.tr.noG2 <- x.tr[, colnames(x.tr) != "G2"]
cv.test <- cv.glmnet(x.tr.noG2, y.tr, alpha = 1, family = "binomial")
min(cv.test$cvm) #test model cv, where we dropped G2 
min(cv.l$cvm) # Original model CV 



blam.1 <- cv.l$lambda.min        # best lambda
blam.1

# Fit final model at chosen lambda
mod.l1 <- glmnet(x = x.tr, y = y.tr,
                 alpha = 1,
                 lambda = blam.1,
                 family = "binomial")

# Which features survived?
coef(mod.l1)
sum(coef(mod.l1) != 0) - 1       # count of non-zero (excl. intercept)


# 6. Predict on the test set
prob.1 <- predict(mod.l1, newx = x.ts, type = "response")
pred.1 <- ifelse(prob.1 > 0.5, 1, 0)

# Confusion matrix
cm.mat <- table(Predicted = pred.1, Actual = d1.ts$pass)
cm.mat

# Accuracy, sensitivity, specificity
TN.m <- cm.mat["0","0"]; FP.m <- cm.mat["1","0"]
FN.m <- cm.mat["0","1"]; TP.m <- cm.mat["1","1"]

acc.mat  <- (TP.m + TN.m) / (TP.m + TN.m + FP.m + FN.m)
sens.mat <- TP.m / (TP.m + FN.m)
spec.mat <- TN.m / (TN.m + FP.m)

acc.mat
sens.mat
spec.mat

# ROC curve and AUC
roc.1 <- roc(response  = as.numeric(as.character(d1.ts$pass)),
             predictor = as.numeric(prob.1),
             quiet = TRUE)

# To plot the AUC curve
plot(roc.1, col = "steelblue", lwd = 2,
     main = "Math - ROC curve")
abline(a = 0, b = 1, lty = 2, col = "grey")
text(0.6, 0.2, paste("AUC =", round(auc(roc.1), 3)), cex = 1.2)

auc.mat <- as.numeric(auc(roc.1))
auc.mat




# PART 2 : PORTUGUESE (d2)


# 1. Inspect 
head(d2)
str(d2)
dim(d2)

# 2. Build binary response 
d2$pass <- ifelse(d2$G3 >= 10, 1, 0)
d2 <- d2[ , !names(d2) %in% "G3"]

table(d2$pass) / nrow(d2)


# 3. Exploratory Data Analysis

# 3a. Class balance
par(mfrow = c(1, 1))
barplot(table(d2$pass),
        names.arg = c("Fail", "Pass"),
        col  = c("tomato", "steelblue"),
        main = "Portuguese - Class balance",
        ylab = "Number of students")

# 3b. Boxplots
par(mfrow = c(2, 2))
boxplot(G1 ~ pass, data = d2,
        main = "G1 by Pass (Por)",
        col = c("tomato", "steelblue"))
boxplot(G2 ~ pass, data = d2,
        main = "G2 by Pass (Por)",
        col = c("tomato", "steelblue"))
boxplot(absences ~ pass, data = d2,
        main = "Absences by Pass (Por)",
        col = c("tomato", "steelblue"))
boxplot(studytime ~ pass, data = d2,
        main = "Study time by Pass (Por)",
        col = c("tomato", "steelblue"))

# 3c. Categorical features vs pass rate
par(mfrow = c(2, 2))
spineplot(as.factor(d2$pass) ~ as.factor(d2$sex),
          main = "Pass rate by sex (Por)",
          col = c("tomato", "steelblue"))
spineplot(as.factor(d2$pass) ~ as.factor(d2$address),
          main = "Pass rate by address (Por)",
          col = c("tomato", "steelblue"))
spineplot(as.factor(d2$pass) ~ as.factor(d2$higher),
          main = "Pass rate by wants higher ed (Por)",
          col = c("tomato", "steelblue"))
spineplot(as.factor(d2$pass) ~ as.factor(d2$schoolsup),
          main = "Pass rate by school support (Por)",
          col = c("tomato", "steelblue"))

# 3d. Past failures vs pass
par(mfrow = c(1, 1))
barplot(prop.table(table(d2$failures, d2$pass), margin = 1),
        beside = TRUE,
        col = c("tomato", "steelblue"),
        legend.text = c("Fail", "Pass"),
        main = "Past failures vs current pass rate (Por)",
        xlab = "Number of past failures",
        ylab = "Proportion")

# 3e. Correlation heatmap
num_cols <- sapply(d2, is.numeric)
cor_mat  <- cor(d2[, num_cols])
corrplot(cor_mat, method = "color", type = "upper",
         tl.cex = 0.7, tl.col = "black",
         title = "Portuguese - correlation of numeric features",
         mar = c(0, 0, 2, 0))


# 4. Train / Test split
d2$pass <- as.factor(d2$pass)

n <- nrow(d2)
set.seed(234)
s.in <- sample(1:n, size = 0.8 * n)

d2.tr <- d2[s.in, ]
d2.ts <- d2[-s.in, ]

x.tr <- model.matrix(pass ~ ., data = d2.tr)[ , -1]
x.ts <- model.matrix(pass ~ ., data = d2.ts)[ , -1]
y.tr <- d2.tr$pass

dim(x.tr)


# 5. Fit LASSO logistic regression
set.seed(234)
cv.l <- cv.glmnet(x = x.tr, y = y.tr,
                  alpha = 1,
                  family = "binomial",
                  nfolds = 10,
                  type.measure = "deviance")

plot(cv.l)
title("Portuguese - LASSO CV", line = 2.5)

blam.2 <- cv.l$lambda.min
blam.2

mod.l2 <- glmnet(x = x.tr, y = y.tr,
                 alpha = 1,
                 lambda = blam.2,
                 family = "binomial")

coef(mod.l2)
sum(coef(mod.l2) != 0) - 1


# 6. Predict on the test set
prob.2 <- predict(mod.l2, newx = x.ts, type = "response")
pred.2 <- ifelse(prob.2 > 0.5, 1, 0)

# Confusion matrix
cm.por <- table(Predicted = pred.2, Actual = d2.ts$pass)
cm.por

# Accuracy, sensitivity, specificity
TN.p <- cm.por["0","0"]; FP.p <- cm.por["1","0"]
FN.p <- cm.por["0","1"]; TP.p <- cm.por["1","1"]

acc.por  <- (TP.p + TN.p) / (TP.p + TN.p + FP.p + FN.p)
sens.por <- TP.p / (TP.p + FN.p)
spec.por <- TN.p / (TN.p + FP.p)

acc.por
sens.por
spec.por

# ROC curve and AUC
roc.2 <- roc(response  = as.numeric(as.character(d2.ts$pass)),
             predictor = as.numeric(prob.2),
             quiet = TRUE)

plot(roc.2, col = "steelblue", lwd = 2,
     main = "Portuguese - ROC curve")
abline(a = 0, b = 1, lty = 2, col = "grey")
text(0.6, 0.2, paste("AUC =", round(auc(roc.2), 3)), cex = 1.2)

auc.por <- as.numeric(auc(roc.2))
auc.por



# FINAL COMPARISON

summary_tab <- data.frame(
  Subject     = c("Math", "Portuguese"),
  Best_Lambda = c(blam.1,   blam.2),
  Accuracy    = round(c(acc.mat,  acc.por),2),
  Sensitivity = round(c(sens.mat, sens.por),2),
  Specificity = round(c(spec.mat, spec.por),2),
  AUC         = round(c(auc.mat,  auc.por),2)
)
print(summary_tab, row.names = FALSE)


# WHY LASSO (over plain logistic / Ridge / Best-Subset)?

#  - The dataset has ~30 features and many one-hot expansions.
#    Best-subset would search 2^30+ models - not practical.
#  - Plain logistic regression overfits when features are correlated
#    (e.g. G1 and G2 carry overlapping information).
#  - Ridge shrinks but never zeroes features -> harder to interpret.
#  - LASSO shrinks AND selects: it tells us WHICH features matter.
