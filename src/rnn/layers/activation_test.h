#ifndef WS_RNN_ACTIVATION_TEST_H
#define WS_RNN_ACTIVATION_TEST_H

#include "./base_grad_test.h"

class ActivationTest : public BaseGradTest {
 protected:
  virtual void SetUp();
  virtual void TearDown();

  ::testing::AssertionResult checkActivation(
      IActivation* activation, int size);
};

#endif //WS_RNN_ACTIVATION_TEST_H
