#pragma once

#include "spatial_vector.hpp"
#include "inertia.hpp"
#include "enoki_algebra.hpp"

template <typename Algebra>
struct Transform {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using Matrix3 = typename Algebra::Matrix3;
  using RigidBodyInertia = ::RigidBodyInertia<Algebra>;
  using ArticulatedBodyInertia = ::ArticulatedBodyInertia<Algebra>;
  typedef ::MotionVector<Algebra> MotionVector;
  typedef ::ForceVector<Algebra> ForceVector;

  Vector3 translation{0.};
  Matrix3 rotation{Matrix3(1.)};

  friend std::ostream &operator<<(std::ostream &os, const Transform &tf) {
    os << "[ translation: " << tf.translation << "  rotation: " << tf.rotation
       << " ]";
    return os;
  }

  TINY_INLINE void set_identity() {
    Algebra::set_zero(translation);
    // set diagonal entries to one, others to zero
    rotation = Matrix3(1.);
  }

  Transform(const Vector3 &translation) : translation(translation) {}
  Transform(const Matrix3 &rotation) : rotation(rotation) {}
  Transform(const Vector3 &translation, const Matrix3 &rotation)
      : translation(translation), rotation(rotation) {}
  Transform(const Scalar &trans_x, const Scalar &trans_y, const Scalar &trans_z)
      : translation(trans_x, trans_y, trans_z) {}

  /**
   * X1*X2 = plx(E1*E2, r2 + E2T*r1)
   */
  Transform operator*(const Transform &t) const {
    /// XXX this is different from Featherstone: we assume transforms are
    /// right-associative
    Transform tr = *this;
    tr.translation += rotation * t.translation;
    tr.rotation *= t.rotation;
    return tr;
  }

  TINY_INLINE Vector3 apply(const Vector3 &point) const {
    return rotation * point + translation;
  }
  TINY_INLINE Vector3 apply_inverse(const Vector3 &point) const {
    return Algebra::transpose(rotation) * (point - translation);
  }

  Transform inverse() const {
    Transform inv;
    inv.rotation = Algebra::transpose(rotation);
    inv.translation = inv.rotation * -translation;
    return inv;
  }

  /**
   * V = mv(w, v)
   * X*V = mv(E*w, E*(v - r x w))
   */
  inline MotionVector apply(const MotionVector &inVec) const {
    MotionVector outVec;

    Vector3 rxw = Algebra::cross(translation, inVec.top);
    Vector3 v_rxw = inVec.bottom - rxw;
    Matrix3 Et = rotation;  // Algebra::transpose(rotation);

    outVec.top = Et * inVec.top;
    outVec.bottom = Et * v_rxw;

    return outVec;
  }

  /**
   * V = mv(w, v)
   * inv(X)*V = mv(ET*w, ET*v + r x (ET*w))
   */
  inline MotionVector apply_inverse(const MotionVector &inVec) const {
    MotionVector outVec;
    Matrix3 Et = Algebra::transpose(rotation);
    outVec.top = Et * inVec.top;
    outVec.bottom = Et * inVec.bottom + Algebra::cross(translation, outVec.top);
    return outVec;
  }

  /**
   * F = fv(n, f)
   * XT*F = fv(ETn + rxETf, ETf)
   */
  inline ForceVector apply(const ForceVector &inVec) const {
    ForceVector outVec;
    Matrix3 Et = Algebra::transpose(rotation);
    outVec.bottom = Et * inVec.bottom;
    outVec.top = Et * inVec.top;
    outVec.top += Algebra::cross(translation, outVec.bottom);

    return outVec;
  }

  /**
   * F = fv(n, f)
   * X^* F = fv(E(n - rxf), Ef)
   */
  inline ForceVector apply_inverse(const ForceVector &inVec) const {
    const Vector3 &n = inVec.top;
    const Vector3 &f = inVec.bottom;
    Matrix3 Et = rotation;  // Algebra::transpose(rotation);
    ForceVector outVec;
    outVec.top = Et * (n - Algebra::cross(translation, f));
    outVec.bottom = Et * f;
    return outVec;
  }

  /**
   * Computes \f$ X^* I X^{-1} \f$.
   */
  inline RigidBodyInertia apply(const RigidBodyInertia &rbi) const {
    RigidBodyInertia result(rbi.mass);
    const Matrix3 rx = Algebra::cross_matrix(translation);
    const Matrix3 E = rotation;
    const Matrix3 Et = Algebra::transpose(rotation);
    // E(I + rx hx + (h - mr)x rx) E^T
    result.inertia =
        E *
        (rbi.inertia + rx * Algebra::cross_matrix(rbi.com) +
         Algebra::cross_matrix(rbi.com - rbi.mass * translation) * rx) *
        Et;
    // E(h - mr)
    result.com = E * (rbi.com - rbi.mass * translation);
    return result;
  }

  /**
   * Computes \f$ X^T I X \f$.
   */
  inline RigidBodyInertia apply_transpose(const RigidBodyInertia &rbi) const {
    RigidBodyInertia result(rbi.mass);
    const Matrix3 E = rotation;
    const Matrix3 Et = Algebra::transpose(rotation);
    // E^T h + mr
    const Vector3 Eth_mr = Et * rbi.com + rbi.mass * translation;
    const Matrix3 rx = Algebra::cross_matrix(translation);
    // E^T I E - rx(E^T h)x - (E^T h + mr)x rx
    result.inertia =
        (Et * rbi.inertia * E - rx * Algebra::cross_matrix(Et * rbi.com) -
         Algebra::cross_matrix(Eth_mr) * rx);
    // E^T h + mr
    result.com = Eth_mr;
    return result;
  }

  /**
   * Computes \f$ X^* I^A X^{-1} \f$.
   */
  inline ArticulatedBodyInertia apply(const ArticulatedBodyInertia &abi) const {
    ArticulatedBodyInertia result;
    const Matrix3 &E = rotation;
    const Matrix3 Et = Algebra::transpose(rotation);
    const Matrix3 rx = Algebra::cross_matrix(translation);
    // H - rx M
    const Matrix3 HrxM = abi.H - rx * abi.M;
    // E (I + rx H^T + (H - rx M) rx) E^T
    result.I = E * (abi.I + rx * Algebra::transpose(abi.H) + HrxM * rx) * Et;
    // E (H - rx M) E^T
    result.H = E * HrxM * Et;
    // E M E^T
    result.M = E * abi.M * Et;
    return result;
  }

  /**
   * Computes \f$ X^T I^A X \f$.
   */
  inline ArticulatedBodyInertia apply_transpose(
      const ArticulatedBodyInertia &abi) const {
    ArticulatedBodyInertia result;
    const Matrix3 E = rotation;
    const Matrix3 Et = Algebra::transpose(rotation);
    const Matrix3 rx = Algebra::cross_matrix(translation);
    // M' = E^T M E
    const Matrix3 Mp = Et * abi.M * E;
    result.M = Mp;
    // H' = E^T H E
    const Matrix3 Hp = Et * abi.H * E;
    // H' + rx M'
    const Matrix3 HrxM = Hp + rx * Mp;
    // E^T I E - rx H'^T - (H' + rx M') rx
    result.I = Et * abi.I * E - rx * Algebra::transpose(Hp) - HrxM * rx;
    // H' + rx M'
    result.H = HrxM;
    return result;
  }

  ENOKI_STRUCT(Transform, translation, rotation)
};
ENOKI_STRUCT_SUPPORT(Transform, translation, rotation)
