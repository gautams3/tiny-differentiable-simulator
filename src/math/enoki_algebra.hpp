#pragma once

#include <enoki/autodiff.h>
#include <enoki/cuda.h>
#include <enoki/dynamic.h>
#include <enoki/quaternion.h>
#include <enoki/special.h>  // for erf()

#include <vector>

#include "spatial_vector.hpp"

struct EnokiAlgebra {
  using Scalar = double;
  using Vector3 = enoki::Array<Scalar, 3>;
  using Vector6 = enoki::Array<Scalar, 6>;
  using VectorX = std::vector<Scalar>;
  using Matrix3 = enoki::Matrix<Scalar, 3>;
  using Matrix6 = enoki::Matrix<Scalar, 6>;
  using Quaternion = enoki::Quaternion<Scalar>;
  using SpatialVector = ::SpatialVector<EnokiAlgebra>;
  using MotionVector = ::MotionVector<EnokiAlgebra>;
  using ForceVector = ::ForceVector<EnokiAlgebra>;

  template <typename T>
  ENOKI_INLINE static auto transpose(const T &matrix) {
    return enoki::transpose(matrix);
  }

  template <typename T>
  ENOKI_INLINE static auto inverse(const T &matrix) {
    return enoki::inverse(matrix);
  }

  template <typename T>
  ENOKI_INLINE static auto inverse_transpose(const T &matrix) {
    return enoki::inverse_transpose(matrix);
  }

  template <typename T1, typename T2>
  ENOKI_INLINE static auto cross(const T1 &vector_a, const T2 &vector_b) {
    return enoki::cross(vector_a, vector_b);
  }

  /**
   * V1 = mv(w1, v1)
   * V2 = mv(w2, v2)
   * V1 x V2 = mv(w1 x w2, w1 x v2 + v1 x w2)
   */
  static inline MotionVector cross(const MotionVector &a,
                                   const MotionVector &b) {
    return MotionVector(
        enoki::cross(a.top, b.top),
        enoki::cross(a.top, b.bottom) + enoki::cross(a.bottom, b.top));
  }

  /**
   * V = mv(w, v)
   * F = fv(n, f)
   * V x* F = fv(w x n + v x f, w x f)
   */
  static inline ForceVector cross(const MotionVector &a, const ForceVector &b) {
    return ForceVector(
        enoki::cross(a.top, b.top) + enoki::cross(a.bottom, b.bottom),
        enoki::cross(a.top, b.bottom));
  }

  /**
   * V = mv(w, v)
   * F = mv(n, f)
   * V.F = w.n + v.f
   */
  ENOKI_INLINE static Scalar dot(const MotionVector &a, const ForceVector &b) {
    return enoki::dot(a.top, b.top) + enoki::dot(a.bottom, b.bottom);
  }
  ENOKI_INLINE static Scalar dot(const ForceVector &a, const MotionVector &b) {
    return dot(b, a);
  }

  template <typename T1, typename T2>
  ENOKI_INLINE static auto dot(const T1 &vector_a, const T2 &vector_b) {
    return enoki::dot(vector_a, vector_b);
  }

  ENOKI_INLINE static Scalar norm(const SpatialVector &v) {
    Vector6 v6 = v;
    return enoki::norm(v6);
  }

  template <typename T>
  ENOKI_INLINE static Scalar norm(const T &v) {
    return enoki::norm(v);
  }

  template <typename T>
  ENOKI_INLINE static auto normalize(T &v) {
    return enoki::normalize(v);
  }

  /**
   * Cross product in matrix form.
   */
  ENOKI_INLINE static Matrix3 cross_matrix(const Vector3 &v) {
    return Matrix3(0., -v[2], v[1], v[2], 0., -v[0], -v[1], v[0], 0.);
  }

  ENOKI_INLINE static Matrix3 zero33() { return Matrix3(0); }
  ENOKI_INLINE static Matrix3 diagonal3(const Vector3 &v) {
    return Matrix3(v[0], 0, 0, 0, v[1], 0, 0, 0, v[2]);
  }
  ENOKI_INLINE static Matrix3 diagonal3(const Scalar &v) { return Matrix3(v); }

  ENOKI_INLINE static Scalar zero() { return 0; }
  ENOKI_INLINE static Scalar one() { return 1; }
  ENOKI_INLINE static Scalar half() { return 0.5; }

  ENOKI_INLINE static Vector3 zero3() { return Vector3(0); }
  ENOKI_INLINE static Vector3 unit3_x() { return Vector3(1, 0, 0); }
  ENOKI_INLINE static Vector3 unit3_y() { return Vector3(0, 1, 0); }
  ENOKI_INLINE static Vector3 unit3_z() { return Vector3(0, 0, 1); }

  template <std::size_t Size1, std::size_t Size2>
  ENOKI_INLINE static void assign_block(
      enoki::Matrix<Scalar, Size1> &output,
      const enoki::Matrix<Scalar, Size2> &input, std::size_t i, std::size_t j,
      std::size_t m = Size2, std::size_t n = Size2, std::size_t input_i = 0,
      std::size_t input_j = 0) {
    assert(i + m <= Size1 && j + n <= Size1);
    assert(input_i + m <= Size2 && input_j + n <= Size2);
    for (std::size_t ii = 0; ii < m; ++ii) {
      for (std::size_t jj = 0; jj < n; ++jj) {
        output(ii + i, jj + j) = input(ii + input_i, jj + input_j);
      }
    }
  }

  ENOKI_INLINE static Matrix3 quat_to_matrix(const Quaternion &quat) {
    return enoki::quat_to_matrix<Matrix3>(quat);
  }
  ENOKI_INLINE static Matrix3 quat_to_matrix(const Scalar &x, const Scalar &y,
                                             const Scalar &z, const Scalar &w) {
    return enoki::quat_to_matrix<Matrix3>(Quaternion(x, y, z, w));
  }
  ENOKI_INLINE static Quaternion matrix_to_quat(const Matrix3 &m) {
    return enoki::matrix_to_quat(m);
  }
  ENOKI_INLINE static Quaternion axis_angle_quaternion(const Vector3 &axis,
                                                       const Scalar &angle) {
    return enoki::rotate<Quaternion, Vector3>(axis, angle);
  }
  ENOKI_INLINE static Matrix3 rotation_x_matrix(const Scalar &angle) {
    Scalar c = enoki::cos(angle);
    Scalar s = enoki::sin(angle);
    return Matrix3(1, 0, 0, 0, c, s, 0, -s, c);
  }

  ENOKI_INLINE static Matrix3 rotation_y_matrix(const Scalar &angle) {
    Scalar c = enoki::cos(angle);
    Scalar s = enoki::sin(angle);
    return Matrix3(c, 0, -s, 0, 1, 0, s, 0, c);
  }

  ENOKI_INLINE static Matrix3 rotation_z_matrix(const Scalar &angle) {
    Scalar c = enoki::cos(angle);
    Scalar s = enoki::sin(angle);
    return Matrix3(c, s, 0, -s, c, 0, 0, 0, 1);
  }

  ENOKI_INLINE static Vector3 rotate(const Quaternion &q, const Vector3 &v) {
    return enoki::quat_to_matrix<Matrix3>(q) * v;
  }

  /**
   * Computes the quaternion delta given current rotation q, angular velocity w,
   * time step dt.
   */
  ENOKI_INLINE static Quaternion quat_velocity(const Quaternion &q,
                                               const Vector3 &w,
                                               const Scalar &dt) {
    Quaternion delta(q[3] * w[0] + q[1] * w[2] - q[2] * w[1],
                     q[3] * w[1] + q[2] * w[0] - q[0] * w[2],
                     q[3] * w[2] + q[0] * w[1] - q[1] * w[0],
                     -q[0] * w[0] - q[1] * w[1] - q[2] * w[2]);
    delta *= 0.5 * dt;
    return delta;
  }

  ENOKI_INLINE static const Scalar &quat_x(const Quaternion &q) { return q[0]; }
  ENOKI_INLINE static const Scalar &quat_y(const Quaternion &q) { return q[1]; }
  ENOKI_INLINE static const Scalar &quat_z(const Quaternion &q) { return q[2]; }
  ENOKI_INLINE static const Scalar &quat_w(const Quaternion &q) { return q[3]; }

  template <std::size_t Size>
  ENOKI_INLINE static void set_zero(enoki::Matrix<Scalar, Size> &m) {
    m = 0;
  }
  template <std::size_t Size>
  ENOKI_INLINE static void set_zero(enoki::Array<Scalar, Size> &m) {
    m = 0;
  }

  ENOKI_INLINE static double to_double(const Scalar &s) {
    return static_cast<double>(s);
  }

  template <typename T>
  static void print(const std::string &title, T object) {
    std::cout << title << "\n" << object << std::endl;
  }

  /**
   * Computes 6x6 matrix by multiplying a and b^T.
   */
  template <std::size_t Size>
  ENOKI_INLINE static enoki::Matrix<Scalar, Size> mul_transpose(
      const enoki::Array<Scalar, Size> &a,
      const enoki::Array<Scalar, Size> &b) {
    enoki::Matrix<Scalar, Size> m(0.0);
    for (std::size_t c = 0; c < Size; ++c) {
      m.col(c) = a * b[c];
    }
    return m;
  }
  ENOKI_INLINE static Matrix6 mul_transpose(const SpatialVector &a,
                                            const SpatialVector &b) {
    Vector6 a6 = a, b6 = b;
    return mul_transpose(a6, b6);
  }

  EnokiAlgebra() = delete;
};
