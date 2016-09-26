#include "layer_test.h"
#include <memory>

namespace {
  const std::vector<const char*> kLayerNames = {
    "sigmoid", "tanh", "scrnfast0", "scrn0", "scrn10", "gru", "gru-insyn", "gru-full"
  };
  const std::vector<int> kStepsRange = {1, 2, 3, 5, 10};
  const int kLayerSize = 50;
  const int kMaxLayerCount = 2;
  const int kMaxSeed = 20;
}

SimpleCriterion::SimpleCriterion(int size) : vector(size) {
  InitUniform(1, &vector);
}

Real SimpleCriterion::Forward(const Ref<const RowMatrix> hidden_states, int steps) const {
  if (steps == 0) {
    return 0;
  }

  return (hidden_states.topRows(steps).array().square().matrix() * vector.transpose()).mean();
}

RowMatrix SimpleCriterion::Backward(const Ref<const RowMatrix> hidden_states, int steps) const {
  RowMatrix hidden_grads = hidden_states;
  hidden_grads.setZero();
  for (int i = 0; i < steps; ++i) {
    hidden_grads.row(i) = vector.cwiseProduct(hidden_states.row(i));
    hidden_grads.row(i) *= 2. / steps;
  }
  return hidden_grads;
}

void LayerTest::SetUp() {
  BaseGradTest::SetUp();
}

void LayerTest::TearDown() {
  BaseGradTest::TearDown();
}

::testing::AssertionResult
LayerTest::checkHiddenLayerOutputToInput(IRecLayer* layer, int size, int steps) {
  std::shared_ptr<IRecUpdater> updater(layer->CreateUpdater());
  SimpleCriterion crit(size);

  auto compute_cost = [&] (const RowMatrix& input) {
    updater->GetInputMatrix().topRows(input.rows()) = input;
    updater->ForwardSequence(input.rows());
    return crit.Forward(updater->GetOutputMatrix(), input.rows());
  };
  auto compute_grads = [&](const RowMatrix& input) {
    updater->GetInputMatrix().topRows(input.rows()) = input;
    updater->ForwardSequence(input.rows());
    updater->GetOutputGradMatrix() = crit.Backward(updater->GetOutputMatrix(), input.rows());
    updater->BackwardSequence(input.rows(), 0, 0, 0);
    return updater->GetInputGradMatrix().topRows(input.rows());
  };

  RowMatrix input(steps, size);
  InitNormal(1, &input);

  return checkGradients<RowMatrix>(input, compute_cost, compute_grads);
}

template<class Matrix, class T1, class T2>
::testing::AssertionResult
LayerTest::checkHiddenLayerSingleWeightGradients(
    IRecLayer* layer, IRecUpdater* updater, const SimpleCriterion& crit,
    Matrix weight, int weight_idx,
    T1 GetWeights, T2 GetGradients,
    int steps) {
  auto compute_cost =
      [layer, updater, steps, crit, GetWeights, GetGradients] (const Matrix& weight) {
        *GetWeights(layer) = weight;
        updater->ForwardSequence(steps);
        return crit.Forward(updater->GetOutputMatrix(), steps);
      };

  auto compute_grads =
      [layer, updater, steps, crit, GetWeights, GetGradients] (const Matrix& weight) {
        *GetWeights(layer) = weight;
        updater->ForwardSequence(steps);
        updater->GetOutputGradMatrix() = crit.Backward(updater->GetOutputMatrix(), steps);
        updater->BackwardSequence(steps, 0, 0, 0);
        return *GetGradients(updater);
      };

  return checkGradients<Matrix>(weight, compute_cost, compute_grads);
}

::testing::AssertionResult
LayerTest::checkHiddenLayerWeightGradients(IRecLayer* layer, int size, int steps) {
  std::shared_ptr<IRecUpdater> updater(layer->CreateUpdater());
  SimpleCriterion crit(size);

  RowMatrix input(steps, size);
  InitNormal(1, &input);
  updater->GetInputMatrix().topRows(input.rows()) = input;

  int matrix_weight_count = layer->GetWeights()->GetMatrices().size();
  for (int weight_idx = 0; weight_idx < matrix_weight_count; ++weight_idx) {
    fprintf(stderr, "\rTesting weight matrix %d of %d\t", weight_idx + 1, matrix_weight_count);
    const RowMatrix& initial_weights = *layer->GetWeights()->GetMatrices()[weight_idx];
    ::testing::AssertionResult result = checkHiddenLayerSingleWeightGradients(
        layer, updater.get(), crit, initial_weights, weight_idx,
        [weight_idx] (IRecLayer* layer) {
          return layer->GetWeights()->GetMatrices()[weight_idx]; },
        [weight_idx] (IRecUpdater* updater) {
          return updater->GetMatrices()[weight_idx]->GetGradients(); },
        steps
    );

    if (!result) {
      return result;
    }
  }

  int vector_weight_count = layer->GetWeights()->GetVectors().size();
  for (int weight_idx = 0; weight_idx < vector_weight_count; ++weight_idx) {
    fprintf(stderr, "\rTesting weight vector %d of %d\t", weight_idx + 1, vector_weight_count);
    RowVector initial_weights = *layer->GetWeights()->GetVectors()[weight_idx];
    ::testing::AssertionResult result = checkHiddenLayerSingleWeightGradients(
        layer, updater.get(), crit, initial_weights, weight_idx,
        [weight_idx] (IRecLayer* layer) {
          return layer->GetWeights()->GetVectors()[weight_idx]; },
        [weight_idx] (IRecUpdater* updater) {
          return updater->GetVectors()[weight_idx]->GetGradients(); },
        steps);

    if (!result) {
      return result;
    }
  }
  return ::testing::AssertionSuccess();
}

TEST_F(LayerTest, LayerConstitution) {
  for (auto hidden_type : kLayerNames) {
    for (int count = 1; count <= kMaxLayerCount; ++count) {
      std::unique_ptr<IRecLayer> layer(CreateLayer(hidden_type, kLayerSize, count));

      // TODO: NULL to nullptr
      ASSERT_TRUE(layer.get() != NULL) << "ERROR create to build the network";

      std::shared_ptr<IRecUpdater> updater(layer->CreateUpdater());

      auto weight_matrices = layer->GetWeights()->GetMatrices();
      auto update_matrices = updater->GetMatrices();
      ASSERT_TRUE(weight_matrices.size() == update_matrices.size())
        << "ERROR Count of weight matrices doesn't match count of updaters: "
        << static_cast<int>(weight_matrices.size())
        << " != "
        << static_cast<int>(update_matrices.size());

      auto weight_vectors = layer->GetWeights()->GetVectors();
      auto update_vectors = updater->GetVectors();
      ASSERT_TRUE(weight_vectors.size() == update_vectors.size())
        << "ERROR Count of weight vectors doesn't match count of updaters: "
        << static_cast<int>(weight_vectors.size())
        << " != "
        << static_cast<int>(update_vectors.size());
    }
  }
}

TEST_F(LayerTest, GradTest) {
  for (auto hidden_type : kLayerNames) {
    for (int count = 1; count <= kMaxLayerCount; ++count) {
      for (int steps : kStepsRange) {
        for (int seed = 0; seed < kMaxSeed; ++seed) {
          srand(seed);
          std::shared_ptr<IRecLayer> layer(CreateLayer(hidden_type, kLayerSize, count));
          fprintf(stderr,
                  "Testing hidden: type=%s, count=%d, size=%d, steps=%d\n",
                  hidden_type, count, kLayerSize, steps);
          EXPECT_TRUE(checkHiddenLayerOutputToInput(layer.get(), kLayerSize, steps));
          EXPECT_TRUE(checkHiddenLayerWeightGradients(layer.get(), kLayerSize, steps));
        }
      }
    }
  }
}
