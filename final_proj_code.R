# Load libraries
library(mvtnorm)
library(mcmcplots)

# Read in data
voters <- read.csv("processed_voter_data.csv", header = TRUE)

# Remove superfluous variables (faminc_didnotsay, pid_other, employment_other, marstat_other, educ_nocollege)
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
fit		<- glm(voted_clinton ~ ., data = voters_train, family = binomial(link = 'logit'))
bhat	<- coef(fit)
vbeta	<- vcov(fit)

### LOGISTIC REGRESSION VIA NORMAL APPROXIMATION ###
# Set number of replicates 
B <- 2000

# Set seed
set.seed(3924)

# Sample beta from normal approximation to the posterior
beta	<- rmvnorm(B, mean = bhat, sigma = vbeta)

# Perform visual diagnostics for convergence on beta0 (assuming others are similar)
beta0 <- as.matrix(beta[,1])
colnames(beta0) <- 'beta[0]'
mcmcplot1(beta0, greek = TRUE)

# Compute mean of simulated betas
mean_beta <- as.matrix(apply(beta, 2, mean))

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

### LOGISTIC REGRESSION VIA METROPOLIS-HASTINGS WITH PRIOR ON BETA OF 1 ###
# Number of replicates
B		<- 40000

# Define function for log of posterior density
tdens	<- function(b, X, y){
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
    r		<- exp(tdens(t(bstar), X, Y)-tdens(beta[t-1,], X, Y)) 
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
r1 <- sample_beta(seed = 324, tau = 0.42, B = 50000)
r2 <- sample_beta(seed = 3984, tau = 0.42, B = 50000)
r3 <- sample_beta(seed = 1579, tau = 0.42, B = 50000)
r4 <- sample_beta(seed = 1034, tau = 0.42, B = 50000)

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

t(apply(beta, 2, quantile, probs = c(0.025, 0.5, 0.975)))

# Create diagnostic plots for beta_0 and beta_7
# Note: there is significant autocorrelation so we have to thin by around 50 to get rid of 
# most of the autocorrelation
B <- 50000
beta0_thinned <- as.matrix(r1$beta[(B/2+1):B,1][seq(from = 1, to = B/2, by = 50)])
colnames(beta0_thinned) <- 'beta[0]'
mcmcplot1(beta0_thinned, greek = TRUE)

beta7_thinned <- as.matrix(r1$beta[(B/2+1):B,8][seq(from = 1, to = B/2, by = 50)])
colnames(beta7_thinned) <- 'beta[7]'
mcmcplot1(beta7_thinned, greek = TRUE)

# Find mean of thinned beta
thinned_beta <- rbind(r1$beta[seq(from = B/2+1, to = B, by = 50),], 
                      r2$beta[seq(from = B/2+1, to = B, by = 50),],
                      r3$beta[seq(from = B/2+1, to = B, by = 50),],
                      r4$beta[seq(from = B/2+1, to = B, by = 50),])
mean_beta <- as.matrix(apply(thinned_beta, 2, mean))

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

### NON-BAYESIAN GLM MODEL ###
# Make predictions with glm
set.seed(324876)
glm_predict <- predict(fit, newdata = voters_test, type = "response")
y_preds_glm <- rbinom(length(glm_predict), 1, glm_predict)

# Determine accuracy of standard glm
mean(y_preds_glm == voters_test$voted_clinton)

# Look at confusion matrix
table(voters_test$voted_clinton, y_preds_glm)
