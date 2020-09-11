#pragma once

namespace tds {
struct EstimationParameter {
  std::string name{"unnamed_param"};
  double value{1.0};
  double minimum{-std::numeric_limits<double>::infinity()};
  double maximum{std::numeric_limits<double>::infinity()};

  // coefficient of L1 regularization for this parameter
  double l1_regularization{0.};

  // coefficient of L2 regularization for this parameter
  double l2_regularization{0.};

  EstimationParameter &operator=(double rhs) {
    value = rhs;
    return *this;
  }
  explicit operator double() const { return value; }

  double random_value() const {
    return minimum + (double(rand()) / RAND_MAX * (maximum - minimum));
  };
};
}  // namespace tds
