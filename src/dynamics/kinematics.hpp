#pragma once

#include "../multi_body.hpp"

/**
 * Implements the first phase in ABA, CRBA and RNEA, that computes the
 * joint and body transforms, velocities and bias forces.
 * Initializes articulated inertia with the local body inertia.
 *
 * Joint positions q must have dimension of dof().
 * Joint velocities qd must have dimension of dof_qd().
 * If no joint velocities qd are given, qd is assumed to be zero.
 * If no joint accelerations qdd are given, qdd is assumed to be zero.
 */
template <typename Algebra>
void MultiBody<Algebra>::forward_kinematics(const VectorX &q, const VectorX &qd,
                                            const VectorX &qdd) const {
  assert(static_cast<int>(q.size()) == dof());
  assert(qd.empty() || static_cast<int>(qd.size()) == dof_qd());
  assert(qdd.empty() || static_cast<int>(qdd.size()) == dof_qd());

  if (is_floating) {
    // update base-world transform from q, and update base velocity from qd
    base_X_world.rotation = Algebra::quat_to_matrix(q[0], q[1], q[2], q[3]);
    base_X_world.translation = Vector3(q[4], q[5], q[6]);
    if (!qd.empty()) {
      base_velocity.top = Vector3(qd[0], qd[1], qd[2]);
      base_velocity.bottom = Vector3(qd[3], qd[4], qd[5]);
    } else {
      base_velocity.set_zero();
    }

    ForceVector I0_mul_v0 = base_rbi * base_velocity;
    base_bias_force =
        Algebra::cross(base_velocity, I0_mul_v0) - base_applied_force;

    base_abi = base_rbi;
  }

  for (int i = 0; i < static_cast<int>(links.size()); i++) {
    const Link &link = links[i];
    int parent = link.parent_index;

    // update joint transforms, joint velocity (if available)
    Scalar q_val = get_q_for_link(q, i);
    Scalar qd_val = get_qd_for_link(qd, i);
    link.jcalc(q_val, qd_val);

    // std::cout << "Link " << i << " transform: " << link.X_parent <<
    // std::endl;

    if (parent >= 0 || is_floating) {
      const Transform &parent_X_world =
          parent >= 0 ? links[parent].X_world : base_X_world;
      link.X_world = parent_X_world * link.X_parent;
      const MotionVector &parentVelocity =
          parent >= 0 ? links[parent].v : base_velocity;
      MotionVector xv = link.X_parent.apply(parentVelocity);
      link.v = xv + link.vJ;
    } else {
      link.X_world = base_X_world * link.X_parent;
      link.v = link.vJ;
    }
    MotionVector v_x_vJ = Algebra::cross(link.v, link.vJ);
    link.c = v_x_vJ /*+link.c_J[i]*/;

    link.abi = link.rbi;
    ForceVector I_mul_v = link.abi * link.v;
    ForceVector f_ext = link.X_world.apply_inverse(link.f_ext);

    // #ifdef NEURAL_SIM
    //       if (i >= 3) {
    //         if constexpr (is_neural_scalar<Scalar, Algebra>::value) {
    //           // Inputs: Position.
    //           link.X_world.translation[0].assign("link/pos/x");
    //           link.X_world.translation[1].assign("link/pos/y");
    //           Scalar link_pos_yaw = Algebra::atan2(
    //               link.X_world.rotation(0, 1), link.X_world.rotation(0,
    //               0));
    //           link_pos_yaw.assign("link/pos/yaw");

    //           // Inputs: Velocity.
    //           link.v[3].assign("link/vel/x");
    //           link.v[4].assign("link/vel/y");
    //           link.v[2].assign("link/vel/yaw");

    //           // Outputs: Applied Force.
    //           f_ext[3].assign("link/external_force/x");
    //           f_ext[4].assign("link/external_force/y");
    //           f_ext[2].assign("link/external_force/yaw");

    //           // Cache the outputs.
    //           f_ext[3].evaluate();
    //           f_ext[4].evaluate();
    //           f_ext[2].evaluate();
    //         }
    //       }
    // #endif

    link.pA = Algebra::cross(link.v, I_mul_v) - f_ext;
#ifdef DEBUG
    Algebra::print("link.abi", link.abi);
    Algebra::print("I_mul_v", I_mul_v);
    Algebra::print("link.pA", link.pA);
#endif
    // compute helper temporary variables for floating-base RNEA
    // const SpatialVector &parent_a =
    //     parent >= 0 ? links[parent].a : base_acceleration;
    // link.a = link.X_parent.apply(parent_a) + v_x_vJ;
    // if (!qdd.empty()) {
    //   link.a += link.S * get_qdd_for_link(qdd, i);
    // }
    // link.f = link.abi * link.a + link.pA;
  }
}