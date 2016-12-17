#include <opencv.hpp>
#include <iostream>
#include <iomanip>

class GMM {

public:
  GMM(int _K, int _d);
  void fit(cv::Mat X, int T);
  void fit(cv::Mat X, double e = 0.01);
  void show_params();

private:
  int K;
  int d;
  cv::Mat pi;
  cv::Mat mu;
  cv::Mat sigma;
  cv::Mat likelihood;
  cv::Mat expectate(cv::Mat X);
  void maximize(cv::Mat X, cv::Mat gamma);
  cv::Mat gaussian(cv::Mat X);
  double _gaussian(cv::Mat x, cv::Mat mu_k, cv::Mat sigma_k);
  cv::Mat posterior_prob(cv::Mat likelihood);
  double loglikelihood(cv::Mat gamma);
  cv::Mat slice(cv::Mat sigma, int k);
};
