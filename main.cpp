#include "gmm.hpp"

cv::Mat create_data(cv::Mat N, cv::Mat mu, cv::Mat sigma) {
  int K = N.cols;
  int d = mu.cols;
  cv::Mat X = cv::Mat::zeros(cv::Mat::ones(1, K, CV_32S).dot(N), d, CV_64F);
  cv::RNG rng(cv::getTickCount());
  for(int k = 0; k < K; k++) {
    int s = k > 0 ? N.at<int>(0, k - 1) : 0;
    rng.fill(X.rowRange(cv::Range(s, s + N.at<int>(0, k))), cv::RNG::NORMAL, cv::Scalar(mu.at<double>(k, 0)), cv::Scalar(sigma.at<double>(k, 0, 0)));
  }
  return X;
}

int main() {
  int K = 2;
  int d = 1;
  cv::Mat N = (cv::Mat_<int>(1, K) << 1500, 2500);
  cv::Mat mu = (cv::Mat_<double>(K, d) << -2.0, 0.0);
  cv::Mat sigma = (cv::Mat_<double>(K, d, d) << 2.0, 1.0);
  cv::Mat X = create_data(N, mu, sigma);

  GMM gmm(K, 1);
  gmm.fit(X);
  gmm.show_params();

  return 0;
}
