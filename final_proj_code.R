# Load libraries
library(mvtnorm)

# Read in data
voters <- read.csv("processed_voter_data.csv", header = TRUE)

# Remove superfluous variables (faminc_didnotsay, pid_other, employment_other, marstat_other, educ_nocollege)
voters <- voters[,-c(2, 8, 11, 16, 19)]

# Separate into test set and training set
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
B		<- 40000

beta		<- matrix(0, nrow = B, ncol = ncol(X))
ar			<- vector('numeric', length = B)

beta[1,]	<- bhat

# Note: I'm not exponentiating in this function because it was coming to 0 
# since we had values like exp(-3000)
# However, below, instead of doing r <- tdens(t(bstar), X, Y)/tdens(beta[t-1,], X, Y),
# I'm doing r <- exp(tdens(t(bstar), X, Y) - tdens(beta[t-1,], X, Y))
tdens	<- function(b, X, y){
  X <- as.matrix(X)
  y <- t(as.matrix(y))
  y%*%(X%*%b) - sum(log(1 + exp(X%*%b)))
}

tau	<- 0.42 # need to tune this

for(t in 2:B){
  # This print statement is only here so that we know this is working at a reasonable pace
  if (t %% 1000 == 0) {
    print(t)
  }
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

# Check acceptance rate
mean(ar)
t(apply(beta, 2, quantile, probs = c(0.025, 0.5, 0.975)))

# Plot diagnostic plots
# Note: there is significant autocorrelation so we have to thin by around 50 to get rid of 
# most of the autocorrelation
library(mcmcplots)

beta0_thinned <- as.matrix(beta[(B/2+1):B,1][seq(from = 1, to = B/2, by = 50)])
colnames(beta0_thinned) <- 'beta[0]'
beta1_thinned <- as.matrix(beta[(B/2+1):B,2][seq(from = 1, to = B/2, by = 50)])
colnames(beta1_thinned) <- 'beta[1]'
beta2_thinned <- as.matrix(beta[(B/2+1):B,3][seq(from = 1, to = B/2, by = 50)])
colnames(beta1_thinned) <- 'beta[2]'
beta3_thinned <- as.matrix(beta[(B/2+1):B,4][seq(from = 1, to = B/2, by = 50)])
colnames(beta1_thinned) <- 'beta[3]'

mcmcplot1(beta0_thinned, greek = TRUE)
mcmcplot1(beta1_thinned, greek = TRUE)
mcmcplot1(beta2_thinned, greek = TRUE)
mcmcplot1(beta3_thinned, greek = TRUE)

# Find mean of thinned beta
thinned_beta <- beta[seq(from = B/2+1, to = B, by = 50),]
mean_beta <- as.matrix(apply(thinned_beta, 2, mean))

# Make predictions on test set
# First find X_test * mean_beta
prod <- as.matrix(X_test) %*% mean_beta

# Find probabilities for each one
probs <- exp(prod)/(1+exp(prod))

# Make predictions
set.seed(12345)
y_preds <- rbinom(length(probs), 1, probs)

# Determine accuracy
mean(voters_test$voted_clinton == y_preds)

# Look at confusion matrix
table(voters_test$voted_clinton, y_preds)

# Make predictions with glm
glm_predict <- predict(fit, newdata = voters_test, type = "response")
glm_binary <- rep(0, 1600)
glm_binary[glm_predict > 0.5] <- 1

# Determine accuracy of standard glm
mean(glm_binary == voters_test$voted_clinton)

# Look at confusion matrix
table(voters_test$voted_clinton, glm_binary)
