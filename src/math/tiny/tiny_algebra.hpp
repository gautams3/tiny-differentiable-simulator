#pragma once

#include <vector>

#include "../spatial_vector.hpp"
#include "tiny_matrix3x3.h"
#include "tiny_matrix6x6.h"
#include "tiny_matrix_x.h"
#include "tiny_quaternion.h"
#include "tiny_vector3.h"
#include "tiny_vector_x.h"

template <typename TinyScalar, typename TinyConstants>
struct TinyAlgebra {
  using Index = int;
  using Scalar = TinyScalar;
  using Vector3 = TinyVector3<TinyScalar, TinyConstants>;
  using VectorX = TinyVectorX<TinyScalar, TinyConstants>;
  using Matrix3 = TinyMatrix3x3<TinyScalar, TinyConstants>;
  using Matrix6 = TinyMatrix6x6<TinyScalar, TinyConstants>;
  using Quaternion = TinyQuaternion<TinyScalar, TinyConstants>;
  using SpatialVector = ::SpatialVector<TinyAlgebra>;
  using MotionVector = ::MotionVector<TinyAlgebra>;
  using ForceVector = ::ForceVector<TinyAlgebra>;

  template <typename T>
  TINY_INLINE static auto transpose(const T &matrix) {
    return matrix.transpose();
  }

  template <typename T>
  TINY_INLINE static auto inverse(const T &matrix) {
    return matrix.inverse();
  }

  template <typename T>
  TINY_INLINE static auto inverse_transpose(const T &matrix) {
    return matrix.inverse().transpose();
  }

  template <typename T1, typename T2>
  TINY_INLINE static auto cross(const T1 &vector_a, const T2 &vector_b) {
    return vector_a.cross(vector_b);
  }

  TINY_INLINE static Index size(const VectorX &v) { return v.m_size; }

  /**
   * V1 = mv(w1, v1)
   * V2 = mv(w2, v2)
   * V1 x V2 = mv(w1 x w2, w1 x v2 + v1 x w2)
   */
  static inline MotionVector cross(const MotionVector &a,
                                   const MotionVector &b) {
    return MotionVector(a.top.cross(b.top),
                        a.top.cross(b.bottom) + a.bottom.cross(b.top));
  }

  /**
   * V = mv(w, v)
   * F = fv(n, f)
   * V x* F = fv(w x n + v x f, w x f)
   */
  static inline ForceVector cross(const MotionVector &a, const ForceVector &b) {
    return ForceVector(a.top.cross(b.top) + a.bottom.cross(b.bottom),
                       a.top.cross(b.bottom));
  }

  /**
   * V = mv(w, v)
   * F = mv(n, f)
   * V.F = w.n + v.f
   */
  TINY_INLINE static Scalar dot(const MotionVector &a, const ForceVector &b) {
    return a.top.dot(b.top) + a.bottom.dot(b.bottom);
  }
  TINY_INLINE static Scalar dot(const ForceVector &a, const MotionVector &b) {
    return dot(b, a);
  }

  template <typename T1, typename T2>
  TINY_INLINE static auto dot(const T1 &vector_a, const T2 &vector_b) {
    return vector_a.dot(vector_b);
  }

  TINY_INLINE static Scalar norm(const MotionVector &v) {
    return TinyConstants::sqrt1(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] +
                                v[3] * v[3] + v[4] * v[4] + v[5] * v[5]);
  }

  TINY_INLINE static Scalar norm(const ForceVector &v) {
    return TinyConstants::sqrt1(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] +
                                v[3] * v[3] + v[4] * v[4] + v[5] * v[5]);
  }
  template <typename T>
  TINY_INLINE static Scalar norm(const T &v) {
    return v.length();
  }

  TINY_INLINE static auto normalize(const Quaternion &q) {
    Scalar ql = q.length();
    return Quaternion(q.x() / ql, q.y() / ql, q.z() / ql, q.w() / ql);
  }
  TINY_INLINE static auto normalize(const Vector3 &v) { return v / v.length(); }
  TINY_INLINE static auto normalize(const VectorX &v) { return v / v.length(); }

  /**
   * Cross product in matrix form.
   */
  TINY_INLINE static Matrix3 cross_matrix(const Vector3 &v) {
    return TinyVectorCrossMatrix(v);
  }

  TINY_INLINE static Matrix3 zero33() {
    const Scalar o = TinyConstants::zero();
    return Matrix3(o, o, o, o, o, o, o, o, o);
  }
  TINY_INLINE static VectorX zerox(Index size) {
    VectorX v(size);
    v.set_zero();
    return v;
  }
  TINY_INLINE static Matrix3 diagonal3(const Vector3 &v) {
    const Scalar o = TinyConstants::zero();
    return Matrix3(v[0], o, o, o, v[1], o, o, o, v[2]);
  }
  TINY_INLINE static Matrix3 diagonal3(const Scalar &v) {
    const Scalar o = TinyConstants::zero();
    return Matrix3(v, o, o, o, v, o, o, o, v);
  }
  TINY_INLINE static Matrix3 eye3() { return diagonal3(TinyConstants::one()); }

  TINY_INLINE static Scalar zero() { return TinyConstants::zero(); }
  TINY_INLINE static Scalar one() { return TinyConstants::one(); }
  TINY_INLINE static Scalar half() { return TinyConstants::half(); }
  TINY_INLINE static Scalar fraction(const Scalar &a, const Scalar &b) {
    return TinyConstants::fraction(a, b);
  }

  TINY_INLINE static Scalar scalar_from_string(const std::string &s) {
    return TinyConstants::scalar_from_string(s);
  }

  TINY_INLINE static Vector3 zero3() { return Vector3::zero(); }
  TINY_INLINE static Vector3 unit3_x() { return Vector3::makeUnitX(); }
  TINY_INLINE static Vector3 unit3_y() { return Vector3::makeUnitY(); }
  TINY_INLINE static Vector3 unit3_z() { return Vector3::makeUnitZ(); }

  TINY_INLINE static void assign_block(Matrix3 &output, const Matrix6 &input, Index i,
                                       Index j, Index m = 3, Index n = 3,
                                       Index input_i = 0, Index input_j = 0) {
    if (input_i == 0) {
      if (input_j == 0) {
        output = input.m_topLeftMat;
      } else {
        output = input.m_topRightMat;
      }
    } else {
      if (input_j == 0) {
        output = input.m_bottomLeftMat;
      } else {
        output = input.m_bottomRightMat;
      }
    }
  }
  TINY_INLINE static void assign_block(Matrix6 &output, const Matrix3 &input, Index i,
                                       Index j, Index m = 3, Index n = 3,
                                       Index input_i = 0, Index input_j = 0) {
    if (i == 0) {
      if (j == 0) {
        output.m_topLeftMat = input;
      } else {
        output.m_topRightMat = input;
      }
    } else {
      if (j == 0) {
        output.m_bottomLeftMat = input;
      } else {
        output.m_bottomRightMat = input;
      }
    }
  }

  TINY_INLINE static void assign_column(Matrix3 &m, Index i, const Vector3 &v) {
    m[i] = v;
  }
  TINY_INLINE static void assign_column(Matrix6 &m, Index i,
                                        const SpatialVector &v) {
    for (int j = 0; j < 6; ++j) {
      m(j, i) = v[j];
    }
  }

  TINY_INLINE static Matrix3 quat_to_matrix(const Quaternion &quat) {
    Matrix3 m;
    m.setRotation(quat);
    return m;
  }
  TINY_INLINE static Matrix3 quat_to_matrix(const Scalar &x, const Scalar &y,
                                            const Scalar &z, const Scalar &w) {
    Matrix3 m;
    m.setRotation(Quaternion(x, y, z, w));
    return m;
  }
  TINY_INLINE static Quaternion matrix_to_quat(const Matrix3 &m) {
    Quaternion quat;
    m.getRotation(quat);
    return quat;
  }
  TINY_INLINE static Quaternion axis_angle_quaternion(const Vector3 &axis,
                                                      const Scalar &angle) {
    Quaternion quat;
    quat.setRotation(axis, angle);
    return quat;
  }
  TINY_INLINE static Matrix3 rotation_x_matrix(const Scalar &angle) {
    Matrix3 m;
    m.set_rotation_x(angle);
    return m;
  }

  TINY_INLINE static Matrix3 rotation_y_matrix(const Scalar &angle) {
    Matrix3 m;
    m.set_rotation_y(angle);
    return m;
  }

  TINY_INLINE static Matrix3 rotation_z_matrix(const Scalar &angle) {
    Matrix3 m;
    m.set_rotation_z(angle);
    return m;
  }

  TINY_INLINE static Matrix3 rotation_zyx_matrix(const Scalar &r,
                                                 const Scalar &p,
                                                 const Scalar &y) {
    Matrix3 m;
    m.setEulerZYX(r, p, y);
    return m;
  }

  TINY_INLINE static Vector3 rotate(const Quaternion &q, const Vector3 &v) {
    return q.rotate(v);
  }

  /**
   * Computes the quaternion delta given current rotation q, angular velocity w,
   * time step dt.
   */
  TINY_INLINE static Quaternion quat_velocity(const Quaternion &q,
                                              const Vector3 &w,
                                              const Scalar &dt) {
    Quaternion delta(q[3] * w[0] + q[1] * w[2] - q[2] * w[1],
                     q[3] * w[1] + q[2] * w[0] - q[0] * w[2],
                     q[3] * w[2] + q[0] * w[1] - q[1] * w[0],
                     -q[0] * w[0] - q[1] * w[1] - q[2] * w[2]);
    delta *= 0.5 * dt;
    return delta;
  }

  TINY_INLINE static const Scalar &quat_x(const Quaternion &q) { return q.x(); }
  TINY_INLINE static const Scalar &quat_y(const Quaternion &q) { return q.y(); }
  TINY_INLINE static const Scalar &quat_z(const Quaternion &q) { return q.z(); }
  TINY_INLINE static const Scalar &quat_w(const Quaternion &q) { return q.w(); }

  TINY_INLINE static void set_zero(Matrix3 &m) { m.set_zero(); }
  template <typename S, typename U,
            template <typename, typename> typename ColumnType>
  TINY_INLINE static void set_zero(TinyMatrixXxX_<S, U, ColumnType> &m) {
    m.set_zero();
  }

  TINY_INLINE static void set_zero(Vector3 &v) { v.set_zero(); }
  TINY_INLINE static void set_zero(VectorX &v) { v.set_zero(); }
  TINY_INLINE static void set_zero(MotionVector &v) {
    v.top.set_zero();
    v.bottom.set_zero();
  }
  TINY_INLINE static void set_zero(ForceVector &v) {
    v.top.set_zero();
    v.bottom.set_zero();
  }

  TINY_INLINE static double to_double(const Scalar &s) {
    return TinyConstants::getDouble(s);
  }

  static void print(const std::string &title, const Scalar& s) {
    printf("%s %.12f\n", title.c_str(), to_double(s));
  }

  template <typename T>
  static void print(const std::string &title, T object) {
    object.print(title.c_str());
  }

  // /**
  //  * Computes 6x6 matrix by multiplying a and b^T.
  //  */
  // TINY_INLINE static Matrix6 mul_transpose(const SpatialVector &a,
  //                                          const SpatialVector &b) {
  //   return Matrix6::vTimesvTranspose(a, b);
  // }

  TINY_INLINE static Scalar sin(const Scalar &s) {
    return TinyConstants::sin1(s);
  }

  TINY_INLINE static Scalar cos(const Scalar &s) {
    return TinyConstants::cos1(s);
  }

  TINY_INLINE static Scalar abs(const Scalar &s) {
    return TinyConstants::abs(s);
  }

  TinyAlgebra() = delete;
};
