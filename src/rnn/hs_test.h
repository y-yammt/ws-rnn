#ifndef WS_RNN_HS_TEST_H
#define WS_RNN_HS_TEST_H

#include "./layers/base_grad_test.h"
#include "./hs.h"

class HSTest : public BaseGradTest {
 protected:
  virtual void SetUp();
  virtual void TearDown();

 public:
  void checkHS(const Vocabulary& vocab, int hidden_size, MaxEnt* maxent);
};

#endif // WS_RNN_HS_TEST_H
