#include "./hs_test.h"

namespace {
  const int kHSArity = 2;
}

void HSTest::SetUp() {
  BaseGradTest::SetUp();
}

void HSTest::TearDown() {
  BaseGradTest::TearDown();
}

void HSTest::checkHS(const Vocabulary& vocab, int hidden_size, MaxEnt* maxent) {
  HSTree* tree = HSTree::CreateHuffmanTree(vocab, hidden_size, kHSArity);

  std::vector<uint64_t> feature_hashes;
  int maxent_order = 0;
  if (maxent != nullptr) {
    maxent_order = rand() % 5;
    for (int i = 0; i < maxent_order; ++i) {
      feature_hashes.push_back(rand() % (maxent->GetHashSize() - vocab.size()));
    }
  }
  int target_word = rand() % vocab.size();

  auto compute_cost =
      [target_word, tree, maxent_order, maxent, feature_hashes] (const RowVector& hidden) {
        RowVector hidden_grad(hidden);
        Real log10prob = tree->PropagateForwardAndBackward(
            true, target_word, feature_hashes.data(), maxent_order,
            0, 0, 0, 0, 10,
            hidden.data(), hidden_grad.data(), maxent);
        return log10prob * log(10);
      };

  auto compute_grads =
      [target_word, tree, maxent_order, maxent, feature_hashes] (const RowVector& hidden) {
        RowVector hidden_grad(hidden);
        hidden_grad.setZero();
        tree->PropagateForwardAndBackward(
            true, target_word, feature_hashes.data(), maxent_order,
            0, 0, 0, 0, 10,
            hidden.data(), hidden_grad.data(), maxent);
        return hidden_grad;
      };

  RowVector hidden(hidden_size);
  InitUniform(1., &hidden);
  EXPECT_TRUE(checkGradients<RowVector>(hidden, compute_cost, compute_grads));
  delete tree;
}

TEST_F(HSTest, HierarchicalSoftmaxWithoutMaxEnt) {
  srand(0);
  const std::vector<int> sizes = {1, 7, 131};
  for (size_t i = 0; i < sizes.size(); ++i) {
    for (int round = 0; round < 10; ++round) {
      checkHS(getVocab(), sizes[i], nullptr);
    }
  }
}

TEST_F(HSTest, HierarchicalSoftmaxWithMaxEnt) {
  srand(0);
  const std::vector<int> sizes = {1, 7, 131};
  MaxEnt maxent;
  uint64_t hash_size = 101234;
  maxent.Init(hash_size);
  for (uint64_t i = 0; i < hash_size; ++i) {
    Real tmp;
    InitUniform(.1, 1, &tmp);
    maxent.UpdateValue(i, 1., tmp, 0);
  }
  for (size_t i = 0; i < sizes.size(); ++i) {
    for (int round = 0; round < 10; ++round) {
      checkHS(getVocab(), sizes[i], &maxent);
    }
  }
}
