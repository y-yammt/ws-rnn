#ifndef WS_RNN_LAYER_TEST_H
#define WS_RNN_LAYER_TEST_H

#include "./base_grad_test.h"
#include "./interface.h"

class SimpleCriterion {
 public:
  explicit SimpleCriterion(int size);
  Real Forward(const Ref<const RowMatrix> hidden_states, int steps) const;
  RowMatrix Backward(const Ref<const RowMatrix> hidden_states, int steps) const;

 protected:
  RowVector vector;
};

class LayerTest : public BaseGradTest {
 protected:
  virtual void SetUp();
  virtual void TearDown();

  ::testing::AssertionResult
  checkHiddenLayerOutputToInput(IRecLayer* layer, int size, int steps);

  template<class Matrix, class T1, class T2>
  ::testing::AssertionResult
  checkHiddenLayerSingleWeightGradients(
      IRecLayer* layer, IRecUpdater* updater, const SimpleCriterion& crit,
      Matrix weight, int weight_idx,
      T1 GetWeights, T2 GetGradients,
      int steps);

  ::testing::AssertionResult
  checkHiddenLayerWeightGradients(IRecLayer* layer, int size, int steps);
};

#endif // WS_RNN_LAYER_TEST_H
