#pragma once

#include "inertia.hpp"
#include "spatial_vector.hpp"

namespace tds {
template <typename Algebra>
struct Transform {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using Matrix3 = typename Algebra::Matrix3;
  using Matrix6 = typename Algebra::Matrix6;
  using RigidBodyInertia = tds::RigidBodyInertia<Algebra>;
  using ArticulatedBodyInertia = tds::ArticulatedBodyInertia<Algebra>;
  typedef tds::MotionVector<Algebra> MotionVector;
  typedef tds::ForceVector<Algebra> ForceVector;

  Vector3 translation{Algebra::zero3()};
  Matrix3 rotation{Algebra::eye3()};

  Transform() = default;
  Transform(const Vector3 &translation) : translation(translation) {}
  Transform(const Matrix3 &rotation) : rotation(rotation) {}
  Transform(const Vector3 &translation, const Matrix3 &rotation)
      : translation(translation), rotation(rotation) {}
  Transform(const Scalar &trans_x, const Scalar &trans_y, const Scalar &trans_z)
      : translation(trans_x, trans_y, trans_z) {}

  friend std::ostream &operator<<(std::ostream &os, const Transform &tf) {
    os << "[ translation: " << tf.translation << "  rotation: " << tf.rotation
       << " ]";
    return os;
  }
  void print(const char *title) const {
    printf("%s\n", title);
    printf("  translation:  %.4f\t%.4f\t%.4f\n",
           Algebra::to_double(translation[0]),
           Algebra::to_double(translation[1]),
           Algebra::to_double(translation[2]));
    printf("  rotation:     %.4f\t%.4f\t%.4f\n",
           Algebra::to_double(rotation(0, 0)),
           Algebra::to_double(rotation(0, 1)),
           Algebra::to_double(rotation(0, 2)));
    printf("                %.4f\t%.4f\t%.4f\n",
           Algebra::to_double(rotation(1, 0)),
           Algebra::to_double(rotation(1, 1)),
           Algebra::to_double(rotation(1, 2)));
    printf("                %.4f\t%.4f\t%.4f\n",
           Algebra::to_double(rotation(2, 0)),
           Algebra::to_double(rotation(2, 1)),
           Algebra::to_double(rotation(2, 2)));
  }

  TINY_INLINE void set_identity() {
    Algebra::set_zero(translation);
    // set diagonal entries to one, others to zero
    rotation = Algebra::eye3();
  }

  Matrix6 matrix() const {
    Matrix6 m;
    Matrix3 mErx = -rotation * Algebra::cross_matrix(translation);
    Algebra::assign_block(m, rotation, 0, 0);
    Algebra::assign_block(m, Algebra::zero33(), 0, 3);
    Algebra::assign_block(m, mErx, 3, 0);
    Algebra::assign_block(m, rotation, 3, 3);
    return m;
  }

  Matrix6 matrix_transpose() const {
    Matrix6 m;
    Matrix3 Et = Algebra::transpose(rotation);
    Matrix3 mErxT =
        Algebra::transpose(-rotation * Algebra::cross_matrix(translation));
    Algebra::assign_block(m, Et, 0, 0);
    Algebra::assign_block(m, mErxT, 0, 3);
    Algebra::assign_block(m, Algebra::zero33(), 3, 0);
    Algebra::assign_block(m, Et, 3, 3);
    return m;
  }

  /**
   * X1*X2 = plx(E1*E2, r2 + E2T*r1)
   */
  // Transform operator*(const Transform &t) const {
  //   Transform tr = *this;
  //   tr.translation = t.translation + t.rotation * translation;
  //   tr.rotation *= t.rotation;
  //   return tr;
  // }
  // Transform operator*(const Transform &t) const {
  //   /// XXX this is different from Featherstone: we assume transforms are
  //   /// right-associative
  //   Transform tr = *this;
  //   tr.translation += rotation * t.translation;
  //   tr.rotation *= t.rotation;
  //   return tr;
  // }
  Transform operator*(const Transform &t) const {
    /// XXX this is different from Featherstone: we assume transforms are
    /// right-associative
    Transform tr = *this;
    // RBDL style
    tr.translation += Algebra::transpose(rotation) * t.translation;
    tr.rotation *= t.rotation;
    return tr;
  }

  TINY_INLINE Vector3 apply(const Vector3 &point) const {
    return Algebra::transpose(rotation) * point + translation;
  }
  TINY_INLINE Vector3 apply_inverse(const Vector3 &point) const {
    // return Algebra::transpose(rotation) * (point - translation);
    return rotation * (point - translation);
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
    const Matrix3 &E = rotation;  // Algebra::transpose(rotation);

    outVec.top = E * inVec.top;
    outVec.bottom = E * v_rxw;

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
    const Matrix3 &E = rotation;
    ForceVector outVec;
    outVec.top = E * (n - Algebra::cross(translation, f));
    outVec.bottom = E * f;
    return outVec;
  }

  /**
   * Computes \f$ X^* I X^{-1} \f$.
   */
  inline RigidBodyInertia apply(const RigidBodyInertia &rbi) const {
    RigidBodyInertia result(rbi.mass);
    const Matrix3 rx = Algebra::cross_matrix(translation);
    const Matrix3 &E = rotation;
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
    const Matrix3 &E = rotation;
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
    // modified version that matches the output of RBDL
    ArticulatedBodyInertia result;
    const Matrix3 E = Algebra::transpose(rotation);  // rotation;
    const Matrix3 Et = rotation;  // Algebra::transpose(rotation);
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
  // inline ArticulatedBodyInertia apply(const ArticulatedBodyInertia &abi)
  // const {
  //   ArticulatedBodyInertia result;
  //   const Matrix3 &E = rotation;
  //   const Matrix3 Et = Algebra::transpose(rotation);
  //   const Matrix3 rx = Algebra::cross_matrix(translation);
  //   // H - rx M
  //   const Matrix3 HrxM = abi.H - rx * abi.M;
  //   // E (I + rx H^T + (H - rx M) rx) E^T
  //   result.I = E * (abi.I + rx * Algebra::transpose(abi.H) + HrxM * rx) * Et;
  //   // E (H - rx M) E^T
  //   result.H = E * HrxM * Et;
  //   // E M E^T
  //   result.M = E * abi.M * Et;
  //   return result;
  // }

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
};
}  // namespace tds
