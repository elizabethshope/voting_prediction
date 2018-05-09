# Load libraries
library(mvtnorm)
library(mcmcplots)

# Read in data
voters <- read.csv("processed_voter_data.csv", header = TRUE)

# Remove superfluous variables (faminc_didnotsay, pid_other, 
# employment_other, marstat_other, educ_nocollege)
voters <- voters[,-c(2, 8, 11, 16, 19)]

# Separate into test set and training set
set.seed(830234)
train_indices <- sample(1:8000, 6400)
voters_train <- voters[train_indices,]
voters_test <- voters[-train_indices,]

# Extract Y, get the length of the data, create model matrix
Y	<- voters_train$voted_clinton
n	<- length(Y)
intercept <- rep(1,n)
X	<- cbind(intercept, voters_train[,c(2:15)])
X_test <- cbind(rep(1, dim(voters_test)[1]), voters_test[,c(2:15)])

# Fit regular glm to get beta_hat and the covariance matrix
fit		<- glm(voted_clinton ~ ., data = voters_train, 
            family = binomial(link = 'logit'))
bhat	<- coef(fit)
vbeta	<- vcov(fit)

### LOGISTIC REGRESSION VIA NORMAL APPROXIMATION ###
# Set number of replicates 
B <- 2000

# Set seed
set.seed(3924)

# Sample beta from normal approximation to the posterior
beta	<- rmvnorm(B, mean = bhat, sigma = vbeta)

# Perform visual diagnostics for convergence on beta0 and beta7
beta0 <- as.matrix(beta[,1])
colnames(beta0) <- 'beta[0]'
mcmcplot1(beta0, greek = TRUE)

beta7 <- as.matrix(beta[,8])
colnames(beta7) <- 'beta[7]'
mcmcplot1(beta7, greek = TRUE)

# Compute mean of simulated betas
mean_beta <- as.matrix(apply(beta, 2, mean))
mean_beta

# First find X_test * mean_beta
Xb <- as.matrix(X_test) %*% mean_beta

# Find probabilities for each one
probs <- exp(Xb)/(1+exp(Xb))

# Make predictions
set.seed(12345)
y_preds <- rbinom(length(probs), 1, probs)

# Determine accuracy
mean(voters_test$voted_clinton == y_preds)

# Look at confusion matrix
table(voters_test$voted_clinton, y_preds)

# Look at 95% credible intervals for interpreting results
coef_cred_int <- t(apply(beta, 2, quantile, probs = c(0.025, 0.5, 0.975)))
rownames(coef_cred_int) <- c("intercept", colnames(voters)[2:15])
round(coef_cred_int, 3)

### LOGISTIC REGRESSION VIA METROPOLIS-HASTINGS WITH PRIOR ON BETA OF 1 ###
# Number of replicates
B		<- 50000

# Define function for log of posterior density
betadens	<- function(b, X, y){
  X <- as.matrix(X)
  y <- t(as.matrix(y))
  y%*%(X%*%b) - sum(log(1 + exp(X%*%b)))
}

# Define function for setting the seed, tuning parameter, and number of samples  
# and outputting the sampled betas and acceptance rate
sample_beta <- function(seed, tau, B) {
  
  # Create empty storage for beta and acceptances
  beta		<- matrix(0, nrow = B, ncol = ncol(X))
  ar			<- vector('numeric', length = B)
  
  # Set initial value of beta
  beta[1,]	<- bhat
  
  # Set seed 
  set.seed(seed)
  
  # Metropolis loop
  for(t in 2:B){
    bstar	<- rmvnorm(1, as.matrix(beta[t-1,]), tau*vbeta)
    r		<- exp(betadens(t(bstar), X, Y)-betadens(beta[t-1,], X, Y)) 
    U		<- runif(1)
    if(U < min(1,r)){
      beta[t,]	<- bstar
      ar[t]		<- 1
    } else {
      beta[t,]	<- beta[t-1,]
      ar[t]		<- 0
    }
  }
  
  # Determine acceptance rate
  acceptance_rate <- mean(ar)
  return(list(beta = beta, acceptance_rate = acceptance_rate))
}

# Create four chains with different seeds
r1 <- sample_beta(seed = 324, tau = 0.42, B = B)
r2 <- sample_beta(seed = 3984, tau = 0.42, B = B)
r3 <- sample_beta(seed = 1579, tau = 0.42, B = B)
r4 <- sample_beta(seed = 1034, tau = 0.42, B = B)

# Confirm all four acceptance rates are around 23%
r1$acceptance_rate
r2$acceptance_rate
r3$acceptance_rate
r4$acceptance_rate

# Confirm convergence with Gelman-Rubin diagnostic
chain1	<- mcmc(r1$beta[(B/2 + 1):B,])
chain2	<- mcmc(r2$beta[(B/2 + 1):B,])
chain3	<- mcmc(r3$beta[(B/2 + 1):B,])
chain4	<- mcmc(r4$beta[(B/2 + 1):B,])

allChains <- mcmc.list(list(chain1, chain2, chain3, chain4)) 

gelman.diag(allChains)

# Create function to get lag 1 ACF values for a given thinning level
get_lag1 <- function(beta, thinning_level) {
  thinned_beta <- beta[seq(from = B/2+1, to = B, by = thinning_level),]
  lags <- acf(thinned_beta, plot = FALSE)$acf
  lag1_vals <- vector('numeric', length = 15)
  for (i in 1:15) {
    lag1_vals[i] <- lags[, , i][2,i]
  }
  return(lag1_vals)
}

# Plot histograms of Lag 1 ACF with different thinning levels
par(mfrow = c(3,2))
hist(get_lag1(r1$beta, 1), main = "Histogram of Lag 1 ACF No Thinning",
     xlab = "Lag 1 ACF")
hist(get_lag1(r1$beta, 5), main = "Histogram of Lag 1 ACF Thin by 5",
     xlab = "Lag 1 ACF")
hist(get_lag1(r1$beta, 10), main = "Histogram of Lag 1 ACF Thin by 10",
     xlab = "Lag 1 ACF")
hist(get_lag1(r1$beta, 20), main = "Histogram of Lag 1 ACF Thin by 20",
     xlab = "Lag 1 ACF")
hist(get_lag1(r1$beta, 40), main = "Histogram of Lag 1 ACF Thin by 40",
     xlab = "Lag 1 ACF")
hist(get_lag1(r1$beta, 50), main = "Histogram of Lag 1 ACF Thin by 50",
     xlab = "Lag 1 ACF")

# Find mean of thinned beta
thinned_beta <- rbind(r1$beta[seq(from = B/2+1, to = B, by = 50),], 
                      r2$beta[seq(from = B/2+1, to = B, by = 50),],
                      r3$beta[seq(from = B/2+1, to = B, by = 50),],
                      r4$beta[seq(from = B/2+1, to = B, by = 50),])
mean_beta <- as.matrix(apply(thinned_beta, 2, mean))
mean_beta

# Make predictions on test set
# First find X_test * mean_beta
Xb <- as.matrix(X_test) %*% mean_beta

# Find probabilities for each one
probs <- exp(Xb)/(1+exp(Xb))

# Make predictions
set.seed(98342)
y_preds_mh1 <- rbinom(length(probs), 1, probs)

# Determine accuracy
mean(voters_test$voted_clinton == y_preds_mh1)

# Look at confusion matrix
table(voters_test$voted_clinton, y_preds_mh1)

# Look at 95% credible intervals for interpreting results
coef_cred_int <- t(apply(thinned_beta, 2, quantile, probs = c(0.025, 0.5, 0.975)))
rownames(coef_cred_int) <- c("intercept", colnames(voters)[2:15])
round(coef_cred_int, 3)


### LOGISTIC REGRESSION VIA METROPOLIS-HASTINGS WITH MVN PRIOR ON BETA ###
# Number of replicates
B		<- 50000

# Define hyperpriors
mu <- as.matrix(rep(0, 15))
sig <- 10^2
sig_inverse <- solve(sig*diag(15))

# Define function for log of posterior density
betadens2	<- function(b, X, y){
  X <- as.matrix(X)
  y <- t(as.matrix(y))
  y%*%(X%*%b) - sum(log(1 + exp(X%*%b))) - 1/2*t(b - mu)%*%sig_inverse%*%(b - mu)
}

# Define function for setting the seed, tuning parameter, and number of samples  
# and outputting the sampled betas and acceptance rate
sample_beta2 <- function(seed, tau, B) {
  
  # Create empty storage for beta and acceptances
  beta		<- matrix(0, nrow = B, ncol = ncol(X))
  ar			<- vector('numeric', length = B)
  
  # Set initial value of beta
  beta[1,]	<- bhat
  
  # Set seed 
  set.seed(seed)
  
  # Metropolis loop
  for(t in 2:B){
    bstar	<- rmvnorm(1, as.matrix(beta[t-1,]), tau*vbeta)
    r		<- exp(betadens2(t(bstar), X, Y)-betadens2(beta[t-1,], X, Y)) 
    U		<- runif(1)
    if(U < min(1,r)){
      beta[t,]	<- bstar
      ar[t]		<- 1
    } else {
      beta[t,]	<- beta[t-1,]
      ar[t]		<- 0
    }
  }
  
  # Determine acceptance rate
  acceptance_rate <- mean(ar)
  return(list(beta = beta, acceptance_rate = acceptance_rate))
}

# Create four chains with different seeds
r5 <- sample_beta2(seed = 729, tau = .4, B = B)
r6 <- sample_beta2(seed = 6492, tau = .4, B = B)
r7 <- sample_beta2(seed = 9, tau = .4, B = B)
r8 <- sample_beta2(seed = 41, tau = .4, B = B)

# Confirm all four acceptance rates are around 23%
r5$acceptance_rate
r6$acceptance_rate
r7$acceptance_rate
r8$acceptance_rate

# Confirm convergence with Gelman-Rubin diagnostic
chain5	<- mcmc(r5$beta[(B/2 + 1):B,])
chain6	<- mcmc(r6$beta[(B/2 + 1):B,])
chain7	<- mcmc(r7$beta[(B/2 + 1):B,])
chain8	<- mcmc(r8$beta[(B/2 + 1):B,])

allChains <- mcmc.list(list(chain5, chain6, chain7, chain8)) 

gelman.diag(allChains)

# Plot histograms of Lag 1 ACF with different thinning levels
par(mfrow = c(3,2))
hist(get_lag1(r5$beta, 1), main = "Histogram of Lag 1 ACF No Thinning",
     xlab = "Lag 1 ACF")
hist(get_lag1(r5$beta, 5), main = "Histogram of Lag 1 ACF Thin by 5",
     xlab = "Lag 1 ACF")
hist(get_lag1(r5$beta, 10), main = "Histogram of Lag 1 ACF Thin by 10",
     xlab = "Lag 1 ACF")
hist(get_lag1(r5$beta, 20), main = "Histogram of Lag 1 ACF Thin by 20",
     xlab = "Lag 1 ACF")
hist(get_lag1(r5$beta, 40), main = "Histogram of Lag 1 ACF Thin by 40",
     xlab = "Lag 1 ACF")
hist(get_lag1(r5$beta, 50), main = "Histogram of Lag 1 ACF Thin by 50",
     xlab = "Lag 1 ACF")

# Find mean of thinned beta
thinned_beta <- rbind(r5$beta[seq(from = B/2+1, to = B, by = 50),], 
                      r6$beta[seq(from = B/2+1, to = B, by = 50),],
                      r7$beta[seq(from = B/2+1, to = B, by = 50),],
                      r8$beta[seq(from = B/2+1, to = B, by = 50),])
mean_beta <- as.matrix(apply(thinned_beta, 2, mean))
mean_beta

# Make predictions on test set
# First find X_test * mean_beta
Xb <- as.matrix(X_test) %*% mean_beta

# Find probabilities for each one
probs <- exp(Xb)/(1+exp(Xb))

# Make predictions
set.seed(372)
y_preds_mh2 <- rbinom(length(probs), 1, probs)

# Determine accuracy
mean(voters_test$voted_clinton == y_preds_mh2)

# Look at confusion matrix
table(voters_test$voted_clinton, y_preds_mh2)

# Look at 95% credible intervals for interpreting results
coef_cred_int <- t(apply(thinned_beta, 2, quantile, probs = c(0.025, 0.5, 0.975)))
rownames(coef_cred_int) <- c("intercept", colnames(voters)[2:15])
round(coef_cred_int, 3)

### NON-BAYESIAN GLM MODEL ###
# Look at model summary
summary(fit)

# Make predictions with glm
set.seed(324876)
glm_predict <- predict(fit, newdata = voters_test, type = "response")
y_preds_glm <- rbinom(length(glm_predict), 1, glm_predict)

# Determine accuracy of standard glm
mean(y_preds_glm == voters_test$voted_clinton)

# Look at confusion matrix
table(voters_test$voted_clinton, y_preds_glm)

# Look at estimated coefficients with 95% confidence intervals
round(cbind(coef(summary(fit))[,c(1)],
            coef(summary(fit))[,c(1)] - 1.96*coef(summary(fit))[,c(2)], 
            coef(summary(fit))[,c(1)] + 1.96*coef(summary(fit))[,c(2)]), 3)
