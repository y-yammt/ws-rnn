#ifndef WS_RNN_BASE_GRAD_TEST_H
#define WS_RNN_BASE_GRAD_TEST_H

#include <functional>
#include <mutex>
#include <gtest/gtest.h>
#include "../words.h"
#include "./activation_functions.h"

class BaseGradTest : public ::testing::Test {
 protected:
  virtual void SetUp();
  virtual void TearDown();

  template<class Matrix>
  ::testing::AssertionResult CheckGradients(
          Matrix &params,
          std::function<Real(const Matrix &)> calc_cost,
          std::function<Matrix(const Matrix &)> calc_grads);

  ::testing::AssertionResult CheckDerivative(
          Real param,
          std::function<Real(Real)> calc_cost,
          std::function<Real(Real)> calc_grads);

  const Vocabulary& GetVocab();

 private:
  static Vocabulary vocab;
};

#endif // WS_RNN_BASE_GRAD_TEST_H
