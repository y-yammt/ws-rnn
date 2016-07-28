#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <numeric>
#include "./util.h"

namespace {
  const unsigned int kRandomSeed = 0;
  const Real kMaxValue = 100;
  const Real kEps = 0.01;
  const size_t kNumValues = 100000;

  std::pair<Real, Real> getMeanVariance(const size_t sz, const Real *values) {
    Real exp_x = 0.0;
    Real exp_squared_x = 0.0;
    for (size_t i = 0; i < sz; ++i) {
      exp_x += values[i] / sz;
      exp_squared_x += values[i] * values[i] / sz;
    }
    return std::pair<Real, Real>(exp_x, exp_squared_x - exp_x * exp_x);
  }
}

TEST(UtilTest, InitUniform) {
  srand(kRandomSeed);
  Real random_values[kNumValues];
  InitUniform(kMaxValue, kNumValues, random_values);
  auto stat = getMeanVariance(kNumValues, random_values);
  EXPECT_NEAR(0.0, stat.first, 1.0);
  EXPECT_NEAR(4.0 * kMaxValue * kMaxValue / 12.0, stat.second,
              2.0 * kMaxValue / sqrt(12.0));
}

TEST(UtilTest, InitNormal) {
  srand(kRandomSeed);
  Real random_values[kNumValues];
  InitNormal(1.0, kNumValues, random_values);
  auto stat = getMeanVariance(kNumValues, random_values);
  EXPECT_NEAR(0.0, stat.first, kEps);
  EXPECT_NEAR(1.0, stat.second, kEps);
}

TEST(UtilTest, Clip) {
  ASSERT_EQ(0.0, Clip(std::numeric_limits<Real>::signaling_NaN(), 1.0));
  ASSERT_EQ(0.0, Clip(std::numeric_limits<Real>::quiet_NaN(), 1.0));
  ASSERT_EQ(-1.0, Clip(-2.0, 1.0));
  ASSERT_EQ(1.0, Clip(2.0, 1.0));
}

TEST(UtilTest, ShrinkMatrix) {
  Eigen::Matrix<Real, 3, 2, Eigen::RowMajor> matrix;
  matrix << std::numeric_limits<Real>::signaling_NaN(),
      std::numeric_limits<Real>::quiet_NaN(),
      -std::numeric_limits<Real>::infinity(),
      std::numeric_limits<Real>::infinity(),
      0,
      kMaxValue;
  ShrinkMatrix(matrix, kMaxValue);
  ASSERT_EQ(0.0, matrix(0, 0));
  ASSERT_EQ(0.0, matrix(0, 1));
  ASSERT_NEAR(-1.0, matrix(1, 0), kEps);
  ASSERT_NEAR(1.0, matrix(1, 1), kEps);
  ASSERT_EQ(0.0, matrix(2, 0));
  ASSERT_LT(0.0, matrix(2, 1));
}

TEST(UtilTest, GetNextRandom) {
  uint64_t state = static_cast<uint64_t>(kRandomSeed);
  Real random_values[kNumValues];
  for (size_t i = 0; i < kNumValues; ++i) {
    random_values[i] = static_cast<Real>(
        GetNextRandom(&state) % static_cast<uint32_t>(kMaxValue + 1));
  }
  auto stat = getMeanVariance(kNumValues, random_values);
  EXPECT_NEAR(kMaxValue / 2, stat.first, 1.0);
  EXPECT_NEAR(kMaxValue * kMaxValue / 12.0, stat.second,
              kMaxValue / sqrt(12.0));
}
