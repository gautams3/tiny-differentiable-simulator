#pragma once

#include "kinematics.hpp"

namespace tds {

/**
 * Composite Rigid Body Algorithm (CRBA) to compute the joint space inertia
 * matrix. M must be a properly initialized square matrix of size dof_qd().
 * The inertia matrix is computed in the base frame.
 */
template <typename Algebra>
void mass_matrix(MultiBody<Algebra> &mb, const typename Algebra::VectorX &q,
                 typename Algebra::MatrixX *M) {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using VectorX = typename Algebra::VectorX;
  using Matrix3 = typename Algebra::Matrix3;
  using Matrix6 = typename Algebra::Matrix6;
  using Quaternion = typename Algebra::Quaternion;
  typedef tds::Transform<Algebra> Transform;
  typedef tds::MotionVector<Algebra> MotionVector;
  typedef tds::ForceVector<Algebra> ForceVector;
  typedef tds::Link<Algebra> Link;
  typedef tds::RigidBodyInertia<Algebra> RigidBodyInertia;
  typedef tds::ArticulatedBodyInertia<Algebra> ArticulatedBodyInertia;

  assert(Algebra::size(q) == mb.dof());
  assert(M != nullptr);
  int n = static_cast<int>(mb.size());
  // printf("n is %i\n", n);
  assert(Algebra::num_rows(*M) == mb.dof_qd());
  assert(Algebra::num_cols(*M) == mb.dof_qd());

  forward_kinematics(mb, q);

  M->set_zero();
  for (int i = n - 1; i >= 0; --i) {
    const Link &link = mb[i];
    int parent = link.parent_index;
    const ArticulatedBodyInertia &Ic = link.abi;
    const Transform &Xp = link.X_parent;
    // ArticulatedBodyInertia delta_I = Xp.apply(Ic);  // shift(Xp, Ic)
    ArticulatedBodyInertia delta_I = Xp.matrix_transpose() * Ic.matrix() * Xp.matrix();
    if (parent >= 0) {
      mb[parent].abi += delta_I;
    } else if (mb.is_floating()) {
      mb.base_abi() += delta_I;
    }
    ForceVector Fi = Ic * link.S;  // Ic.mul_inv(link.S);
    int qd_i = link.qd_index;
    if (link.joint_type == JOINT_FIXED) continue;

    (*M)(qd_i, qd_i) = Algebra::dot(link.S, Fi);

    int j = i;
    while (mb[j].parent_index != -1) {
      Fi = mb[j].X_parent.apply_inverse(Fi);
      j = mb[j].parent_index;
      if (mb[j].joint_type == JOINT_FIXED) continue;
      int qd_j = mb[j].qd_index;
      (*M)(qd_i, qd_j) = Algebra::dot(Fi, mb[j].S);
      (*M)(qd_j, qd_i) = (*M)(qd_i, qd_j);
    }

    if (mb.is_floating()) {
      Fi = mb[j].X_parent.apply_inverse(Fi);
      Algebra::assign_column(*M, qd_i, Fi);
      Algebra::assign_row(*M, qd_i, Fi);
    }
  }
  if (mb.is_floating()) {
    // assign Ic_0 to M(0:6, 0:6)
    M->assign_matrix(0, 0, mb.base_abi().I);
    M->assign_matrix(0, 3, mb.base_abi().H);
    M->assign_matrix(3, 0, Algebra::transpose(mb.base_abi().H));
    M->assign_matrix(3, 3, mb.base_abi().M);
  }
}

template <typename Algebra>
void mass_matrix(MultiBody<Algebra> &mb, typename Algebra::MatrixX *M) {
  mass_matrix(mb, mb.q(), M);
}
}  // namespace tds