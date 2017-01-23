#include "./base_grad_test.h"

namespace {
#ifdef USE_DOUBLES
static const Real kArgEps = 1e-4;
static const Real kToleranceEps = 1e-5;
#else
static const Real kArgEps = 1e-3;
static const Real kToleranceEps = 2e-3;
#endif
}

Vocabulary BaseGradTest::vocab = Vocabulary();

void BaseGradTest::SetUp() {
  static std::once_flag init_flag;
  std::call_once(init_flag, [] {
    std::vector<WordFreq> wordfreqs = {
        WordFreq("</s>", 30301028), WordFreq("the", 35936573),
        WordFreq(",", 35089484), WordFreq(".", 29969612),
        WordFreq("to", 18123964), WordFreq("of", 17337973),
        WordFreq("and", 15726613), WordFreq("a", 15501959),
        WordFreq("in", 13893144), WordFreq("\"", 8916641),
        WordFreq("'s", 6885333), WordFreq("that", 6653833),
        WordFreq("for", 6509312), WordFreq("on", 5696332),
        WordFreq("is", 5622298), WordFreq("The", 5264636),
        WordFreq("was", 4684600), WordFreq("with", 4508758),
        WordFreq("said", 4300819), WordFreq("as", 3726373),
        WordFreq("at", 3625133), WordFreq("it", 3461384),
        WordFreq("by", 3393957), WordFreq("from", 3219386),
        WordFreq("be", 3071588), WordFreq("have", 2998425),
        WordFreq("he", 2996141), WordFreq("has", 2992685),
        WordFreq("his", 2877244), WordFreq("are", 2803481),
        WordFreq("an", 2442763), WordFreq(")", 2251007),
        WordFreq("not", 2240622), WordFreq("(", 2233604),
        WordFreq("will", 2154129), WordFreq("who", 2070074),
        WordFreq("I", 2023836), WordFreq("had", 1982724),
        WordFreq("<unk>", 1947127), WordFreq("their", 1903927),
        WordFreq("--", 1870825), WordFreq("were", 1850652),
        WordFreq("they", 1830186), WordFreq("but", 1800921),
        WordFreq("been", 1743268), WordFreq("this", 1730945),
        WordFreq("which", 1645232), WordFreq("more", 1644514),
        WordFreq("or", 1628689), WordFreq("its", 1587886),
        WordFreq("would", 1520675), WordFreq("about", 1462078),
        WordFreq(":", 1353262), WordFreq("after", 1335023),
        WordFreq("up", 1288758), WordFreq("$", 1273718),
        WordFreq("one", 1263443), WordFreq("than", 1246814),
        WordFreq("also", 1180893), WordFreq("'t", 1170447),
        WordFreq("out", 1169162), WordFreq("her", 1160568),
        WordFreq("you", 1123526), WordFreq("year", 1107231),
        WordFreq("when", 1095691), WordFreq("It", 1095544),
        WordFreq("two", 1024991), WordFreq("people", 1022273),
        WordFreq("-", 990309), WordFreq("all", 986568),
        WordFreq("can", 985453), WordFreq("over", 983773),
        WordFreq("last", 981640), WordFreq("first", 978520),
        WordFreq("But", 971547), WordFreq("into", 966639),
        WordFreq("'", 946117), WordFreq("He", 945595),
        WordFreq("A", 940632), WordFreq("we", 935073),
        WordFreq("In", 923731), WordFreq("she", 899680),
        WordFreq("other", 899409), WordFreq("new", 898716),
        WordFreq("years", 866505), WordFreq("could", 859446),
        WordFreq("there", 834424), WordFreq("?", 826596),
        WordFreq("time", 820922), WordFreq("some", 796911),
        WordFreq("them", 737934), WordFreq("if", 721283),
        WordFreq("no", 719095), WordFreq("percent", 702938),
        WordFreq("so", 688777), WordFreq("what", 687137),
        WordFreq("only", 683531), WordFreq("government", 676017),
        WordFreq("million", 667284), WordFreq("just", 648826)
    };
    vocab.Load(wordfreqs);
  });
}

void BaseGradTest::TearDown() {
}

template<class Matrix>
::testing::AssertionResult BaseGradTest::CheckGradients(
        Matrix &params,
        std::function<Real(const Matrix &)> calc_cost,
        std::function<Matrix(const Matrix &)> calc_grads) {
  Matrix analytical_grads = calc_grads(params);

  for (int r = 0; r < params.rows(); ++r) {
    for (int c = 0; c < params.cols(); ++c) {
      double analytical_grad = analytical_grads(r, c);
      double orig_value = params(r, c);

      params(r, c) = orig_value + kArgEps;
      double cost_plus = calc_cost(params);
      params(r, c) = orig_value - kArgEps;
      double cost_minus = calc_cost(params);
      double numerical_grad = (cost_plus - cost_minus) / (2. * kArgEps);
      params(r, c) = orig_value;

      if (std::abs(analytical_grad - numerical_grad) > kToleranceEps
          || isnan(analytical_grad) || isnan(numerical_grad)) {
        return ::testing::AssertionFailure()
            << params(0, 0) << " " << analytical_grads(0, 0)
            << " (" << cost_plus << " - " << cost_minus << ")\n"
            << "ERROR: numerical gradient differs from analytical"
            << " at pos " << r << "," << c
            << ": (numerical != analytical) "
            << numerical_grad << " != " << analytical_grad
            << " (tolerance: "<< kToleranceEps << ")\n";
      }
    }
  }
  return ::testing::AssertionSuccess();
}

template
::testing::AssertionResult BaseGradTest::CheckGradients<RowVector>(
        RowVector &params,
        std::function<Real(const RowVector &)> calc_cost,
        std::function<RowVector(const RowVector &)> calc_grads);

template
::testing::AssertionResult BaseGradTest::CheckGradients<RowMatrix>(
        RowMatrix &params,
        std::function<Real(const RowMatrix &)> calc_cost,
        std::function<RowMatrix(const RowMatrix &)> calc_grads);

::testing::AssertionResult BaseGradTest::CheckDerivative(
        Real param,
        std::function<Real(Real)> calc_cost,
        std::function<Real(Real)> calc_grads) {
  typedef Eigen::Matrix<Real, 1, 1> ScalarMatrix;
  ScalarMatrix param_as_matrix = ScalarMatrix::Constant(param);
  return CheckGradients<ScalarMatrix>(
          param_as_matrix,
          [&calc_cost](const ScalarMatrix &in) { return calc_cost(in(0)); },
          [&calc_grads](const ScalarMatrix &in) { return ScalarMatrix::Constant(calc_grads(in(0))); }
  );
}

const Vocabulary& BaseGradTest::GetVocab() {
  return BaseGradTest::vocab;
}
