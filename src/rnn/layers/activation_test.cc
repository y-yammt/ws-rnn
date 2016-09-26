#include "activation_test.h"
#include <memory>

namespace {
using ActivationFactory = std::pair<std::string, std::function<IActivation*()>>;

std::vector<ActivationFactory> kActivationFactories = {
    ActivationFactory("sigmoid", [] { return new SigmoidActivation(); } ),
    ActivationFactory("tanh", [] { return new TanhActivation(); } ),
    ActivationFactory("relu", [] { return new ReLUActivation(); } ),
    ActivationFactory("relu-trunc", [] { return new TruncatedReLUActivation(); } )
};
}

void ActivationTest::SetUp() {
  BaseGradTest::SetUp();
}

void ActivationTest::TearDown() {
  BaseGradTest::TearDown();
}

::testing::AssertionResult ActivationTest::checkActivation(
    IActivation* activation, int size) {
  auto compute_cost = [activation] (Real input) {
    activation->Forward(&input, 1);
    return input;
  };
  auto compute_grads = [&](Real input) {
    Real grad = 1;
    activation->Forward(&input, 1);
    activation->Backward(&input, 1, &grad);
    return grad;
  };

  RowVector x(size);
  InitNormal(10, &x);
  for (int i = 0; i < size; ++i) {
    ::testing::AssertionResult result = checkDerivative(x[i], compute_cost, compute_grads);
    if (!result) {
      return result;
    }
  }
  return ::testing::AssertionSuccess();
}

TEST_F(ActivationTest, GradTest) {
  for (auto& pair : kActivationFactories) {
    srand(0);
    std::shared_ptr<IActivation> activation(pair.second());
    EXPECT_TRUE(checkActivation(activation.get(), 1000));
  }
}
