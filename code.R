# install.packages(c("rjags","coda","ggplot2","matrixStats","readr"))
library(rjags)
library(coda)
library(ggplot2)
library(matrixStats)
library(readr)
library(zoo)
library(xtable)

# Load data
df <- read.zoo("data//NP_reduced.csv", header = TRUE, sep = ",", format = "%Y-%m-%d", aggregate = mean)
data <- as.numeric(df)
dates <- index(df)
n <- length(data)

# Build lag matrix for AR(28)

median <- median(data)
mad <- mad(data)
cat("median: ",median, "\nMAD: ",mad)
scaled_data = (data-median)/mad
transformed_data = log(scaled_data+sqrt(1+scaled_data*scaled_data))

par(mfrow=c(2,1))
astsa::tsplot(data)
astsa::tsplot(transformed_data)

p <- 28
T0 <- n - p
Y <- transformed_data[(p+1):n]
X <- matrix(NA, nrow=T0, ncol=p)
for(j in 1:p) X[,j] <- transformed_data[(p+1-j):(n-j)]
X <- as.matrix(X)
# intercept column handled inside JAGS as beta0
n_train = floor(n*(1/5))
test_indices = c(1:n_train)
test_X = X[test_indices,]
X = X[-test_indices,]
test_Y = Y[test_indices]
Y = Y[-test_indices]
n = n-n_train
# JAGS model (Bayesian LASSO hierarchical)
jags_model_string <- "
model {
  # Likelihood
  for(t in 1:T) {
    mu[t] <- beta0 + inprod(beta[1:P], X[t,])
    y[t] ~ dnorm(mu[t], tau)   # tau = 1/sigma2
  }

  # Priors for coefficients (scale mixture -> double exponential)
  for(j in 1:P) {
    beta[j] ~ dnorm(0, 1.0 / lambda2_tau[j])   # precision = 1/tau_j
    lambda2_tau[j] ~ dexp(lambda^2 / 2.0)      # mixing for Laplace
  }

  # Intercept
  beta0 ~ dnorm(0.0, 1.0E-6)

  # Noise
  tau ~ dgamma(0.001, 0.001)    # tau = 1/sigma2
  sigma2 <- 1.0 / tau

  # Hyperprior for lambda (penalty). gamma prior
  lambda ~ dgamma(a_lambda, b_lambda)
  # store lambda^2 for convenience
  lambda_sq <- lambda * lambda
}
"

# Data list for JAGS
data_jags <- list(
  T = nrow(X),
  P = p,
  y = as.numeric(Y),
  X = X,
  a_lambda = 1.0,   # weakly informative; change if you want stronger prior
  b_lambda = 1.0
)

# initial values (function)
inits_fn <- function() {
  list(
    beta = rnorm(p,0,0.1),
    beta0 = mean(Y),
    tau = 1/var(Y),
    lambda = rgamma(1,1,1),
    lambda2_tau = rexp(p, rate = 1.0)
  )
}

# JAGS run settings
n.chains <- 3
n.adapt <- 2000
n.burn <- 5000
n.iter <- 20000
n.thin <- 5

# initialize and run
cat("Compiling JAGS model...\n")
jags_mod <- jags.model(textConnection(jags_model_string),
                       data = data_jags,
                       inits = inits_fn,
                       n.chains = n.chains,
                       n.adapt = n.adapt)

cat("Updating (burn-in) ...\n")
update(jags_mod, n.burn)

params <- c("beta0","beta","sigma2","lambda")
cat("Sampling...\n")
draws <- coda.samples(jags_mod, variable.names = params, n.iter = n.iter, thin = n.thin)

# Combine chains
mcmc_comb <- as.mcmc(do.call(rbind, draws))

# Posterior summaries
summary_stats <- summary(draws)
print(summary_stats)

# Extract beta draws matrix (samples x p)
beta_names <- paste0("beta[",1:p,"]")
beta_draws <- do.call(rbind, lapply(draws, function(ch) as.matrix(ch[,beta_names])))
beta0_draws <- do.call(rbind, lapply(draws, function(ch) as.matrix(ch[,"beta0"])))
lambda_draws <- do.call(rbind, lapply(draws, function(ch) as.matrix(ch[,"lambda"])))
sigma2_draws <- do.call(rbind, lapply(draws, function(ch) as.matrix(ch[,"sigma2"])))

# Posterior means and 95% CI for betas
post_mean_beta <- colMeans(beta_draws)
post_ci_beta <- t(apply(beta_draws, 2, quantile, probs=c(0.025,0.5,0.975)))

# Penalty lambda posterior
lambda_mean <- mean(lambda_draws)
lambda_ci <- quantile(lambda_draws, c(0.025,0.975))

# sigma2 posterior
sigma2_mean <- mean(sigma2_draws)
sigma2_ci <- quantile(sigma2_draws, c(0.025,0.975))


# Table of % times each predictor is (near) zero
eps <- 1e-4   # threshold: |beta| < eps considered 'zero' - changeable
pct_zero <- 100 * colMeans(abs(beta_draws) < eps)

# Save summary table
beta_table <- data.frame(
  lag = 1:p,
  mean = post_mean_beta,
  ci025 = post_ci_beta[,1],
  median = post_ci_beta[,2],
  ci975 = post_ci_beta[,3],
  pct_near_zero = pct_zero
)
print(beta_table)

# Plots: trace, cumulative mean, density for each coefficient + lambda
# Trace + density for a subset (first 6 betas) and lambda
plot_dir <- "plots"
if(!dir.exists(plot_dir)) dir.create(plot_dir)

# Use coda traceplots
for(j in 1:p) {
  png(sprintf("%s/trace_beta%02d.png", plot_dir, j), width=800, height=400)
  traceplot(draws[,paste0("beta[",j,"]")], main=paste("Trace plot beta",j))
  dev.off()
  png(sprintf("%s/density_beta%02d.png", plot_dir, j), width=800, height=400)
  dens <- density(beta_draws[,j])
  plot(dens, main=paste("Posterior density beta",j))
  abline(v=mean(beta_draws[,j]), col="blue")
  dev.off()
  png(sprintf("%s/cummean_beta%02d.png", plot_dir, j), width=800, height=400)
  cm <- cumsum(beta_draws[,j]) / seq_along(beta_draws[,j])
  plot(cm, type='l', main=paste("Cumulative mean beta",j), ylab="cummean")
  abline(h=mean(beta_draws[,j]), col="blue")
  dev.off()
}
# lambda plots
png(sprintf("%s/trace_lambda.png", plot_dir), width=800, height=400)
traceplot(draws[,"lambda"], main="Trace plot lambda")
dev.off()
png(sprintf("%s/density_lambda.png", plot_dir), width=800, height=400)
plot(density(lambda_draws), main="Posterior density lambda")
abline(v=lambda_mean, col="blue")
dev.off()
png(sprintf("%s/cummean_lambda.png", plot_dir), width=800, height=400)
cm_l <- cumsum(lambda_draws)/seq_along(lambda_draws)
plot(cm_l, type='l', main="Cumulative mean lambda")
abline(h=lambda_mean, col="blue")
dev.off()

# Posterior predictive check & one-step ahead prediction (example)
# Posterior predictive for training set: compute fitted mean and predictive intervals
S <- nrow(beta_draws)
fitted_mat <- test_X %*% t(beta_draws) + matrix(rep(beta0_draws, each=nrow(test_X)), nrow=nrow(test_X), byrow=FALSE)
# but simpler: get predictive mean for y
pred_mean_y <- rowMeans(fitted_mat)
pred_ci_lower <- apply(fitted_mat, 1, quantile, 0.025)
pred_ci_upper <- apply(fitted_mat, 1, quantile, 0.975)
Metrics::mape(test_Y,pred_mean_y)


par(mfrow=c(1,1))
astsa::tsplot(test_Y)
lines(pred_ci_lower,col='red',lty=2)
lines(pred_mean_y,col='blue',lty=3)
lines(pred_ci_upper,col='green',lty=4)
legend("topright", 
       legend = c("True Y", "Predictive Lower CI", "Predictive Mean", "Predictive Upper CI"), # Labels for the legend
       col = c("black","red", "blue","green"),
       lty = c(1, 2,3,4))

test_df <- data.frame(
  date = dates[test_indices],
  true_y = test_Y,
  fit_mean = pred_mean_y,
  fit_low = pred_ci_lower,
  fit_high = pred_ci_upper
)

# Save outputs
write.csv(beta_table, "beta_posterior_table.csv", row.names=FALSE)
print(xtable(beta_table, type = "latex"), file = "beta_posterior_table.tex")
save(draws, file="jags_draws.RData")
cat("Finished. Plots are in folder:", plot_dir, "\n")
