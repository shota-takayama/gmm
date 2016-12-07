#include "gmm.hpp"

// constructor
GMM::GMM(int _K, int _d) {
  // declaration
  K = _K;
  d = _d;
  pi = cv::Mat::ones(1, K, CV_64F) / K;
  mu = cv::Mat::zeros(K, d, CV_64F);
  sigma = cv::Mat::zeros(3, (int[]){K, d, d}, CV_64F);

  // filled with random values
  cv::RNG rng(cv::getTickCount());
  rng.fill(mu, cv::RNG::NORMAL, cv::Scalar(0.0), cv::Scalar(1.0));
  rng.fill(sigma, cv::RNG::NORMAL, cv::Scalar(0.0), cv::Scalar(1.0));
  sigma = cv::abs(sigma);
}


// estimate parameters: pi, mu, sigma
void GMM::fit(cv::Mat X, double e) {
  int N = X.rows;
  double Q = -DBL_MAX;
  double delta = e;
  while(delta >= e || delta < 0.0) {
    cv::Mat gamma = expectate(X);
    maximize(X, gamma);
    double Q_hat = 0.0;
    for(int i = 0; i < N; i++) {
      for(int k = 0; k < K; k++) {
        Q_hat += gamma.at<double>(i, k) * std::log(pi.at<double>(0, k) * likelihood.at<double>(i, k) + 0.00000000001);
      }
    }
    std::cout << std::fixed << std::setprecision(-std::log(e)) << Q_hat << std::endl;
    delta = Q_hat - Q;
    Q = Q_hat;
  }
  std::cout << std::endl;
}


void GMM::show_params() {
  std::cout << "pi" << std::endl;
  std::cout << pi << std::endl;
  std::cout << "mu" << std::endl;
  std::cout << mu << std::endl;
  std::cout << "sigma" << std::endl;
  for(int k = 0; k < K; k++) {
    std::cout << slice(sigma, k) << std::endl;
  }
}


// E-step
cv::Mat GMM::expectate(cv::Mat X) {
  likelihood = gaussian(X);
  cv::Mat gamma = posterior_prob(likelihood);
  return gamma;
}


// M-step
void GMM::maximize(cv::Mat X, cv::Mat gamma) {
  int N = X.rows;
  cv::Mat pi_hat = cv::Mat::zeros(1, K, CV_64F);
  cv::Mat mu_hat = cv::Mat::zeros(K, d, CV_64F);
  cv::Mat sigma_hat = cv::Mat::zeros(3, (int[]){K, d, d}, CV_64F);

  // estimate
  for(int k = 0; k < K; k++) {
    double N_k = cv::Mat::ones(N, 1, CV_64F).dot(gamma.col(k));
    for(int i = 0; i < N; i++) {
      cv::Mat _x = X.row(i) - mu.row(k);
      pi_hat.at<double>(0, k) = N_k / N;
      mu_hat.row(k) += gamma.at<double>(i, k) * X.row(i) / N_k;
      slice(sigma_hat, k) += gamma.at<double>(i, k) * _x.t() * _x / N_k;
    }
  }

  // update
  pi = pi_hat.clone();
  mu = mu_hat.clone();
  sigma = sigma_hat.clone();
}


// divide 3d Mat into bunches of 2d Mat
cv::Mat GMM::slice(cv::Mat sigma, int k) {
  double* data = (double*)sigma.data + k * d * d;
  return cv::Mat(2, (int[]){d, d}, CV_64F, data);
}


cv::Mat GMM::posterior_prob(cv::Mat likelihood) {
  int N = likelihood.rows;
  cv::Mat gamma = cv::Mat::zeros(likelihood.size(), CV_64F);
  for(int i = 0; i < N; i++) {
    double sum = pi.row(0).dot(likelihood.row(i));
    gamma.row(i) = pi.row(0).mul(likelihood.row(i)) / sum;
  }
  return gamma;
}


// calculate likelihood Mat
cv::Mat GMM::gaussian(cv::Mat X) {
  int N = X.rows;
  cv::Mat likelihood = cv::Mat::zeros(N, K, CV_64F);
  for(int i = 0; i < N; i++) {
    for(int k = 0; k < K; k++) {
      likelihood.at<double>(i, k) = _gaussian(X.row(i), mu.row(k), slice(sigma, k));
    }
  }
  return likelihood;
}


// calculate likelihood
double GMM::_gaussian(cv::Mat x, cv::Mat mu_k, cv::Mat sigma_k) {
  cv::Mat _x = x - mu_k;
  double numer = std::exp(cv::Mat(-0.5 * _x * sigma_k.inv(cv::DECOMP_SVD) * _x.t()).at<double>(0));
  double denom = std::pow(2 * CV_PI, 1.0 / d) * std::sqrt(cv::determinant(sigma_k));
  return numer / denom;
}
