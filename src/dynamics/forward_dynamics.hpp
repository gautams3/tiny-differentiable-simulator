#pragma once

#include "../multi_body.hpp"

template <typename Algebra>
void MultiBody<Algebra>::forward_dynamics(const VectorX &q, const VectorX &qd,
                                          const VectorX &tau,
                                          const Vector3 &gravity,
                                          VectorX &qdd) const {
  assert(static_cast<int>(q.size()) == dof());
  assert(static_cast<int>(qd.size()) == dof_qd());
  assert(static_cast<int>(qdd.size()) == dof_qd());
  assert(static_cast<int>(tau.size()) == dof_actuated());

  MotionVector spatial_gravity;
  spatial_gravity.bottom = gravity;

  // #ifdef NEURAL_SIM
  //     for (int i = 0; i < dof(); ++i) {
  //       NEURAL_ASSIGN(q[i], "q_" + std::to_string(i));
  //     }
  //     for (int i = 0; i < dof_qd(); ++i) {
  //       NEURAL_ASSIGN(qd[i], "qd_" + std::to_string(i));
  //     }
  // #endif

  forward_kinematics(q, qd);

  for (int i = static_cast<int>(links.size()) - 1; i >= 0; i--) {
    const Link &link = links[i];
    int parent = link.parent_index;
    link.U = link.abi * link.S;
    // std::cout << "link.abi.matrix() * link.S:\n" << link.abi.matrix() *
    // link.S << std::endl; std::cout << "link.abi * link.S:\n" << link.abi *
    // link.S << std::endl; std::cout << "\n\n";
    link.d = Algebra::dot(link.S, link.U);
    Scalar tau_val = get_tau_for_link(tau, i);
    // apply linear joint stiffness and damping
    // see Eqns. (2.76), (2.78) in Rigid Body Dynamics Notes
    // by Shinjiro Sueda
    // https://github.com/sueda/redmax/blob/master/notes.pdf
    // TODO consider nonzero resting position of joint for stiffness?
    tau_val -= link.stiffness * get_q_for_link(q, i);
    tau_val -= link.damping * get_qd_for_link(qd, i);

    // #ifdef NEURAL_SIM
    //       NEURAL_ASSIGN(tau_val, "tau_" + std::to_string(i));
    // #endif

    link.u = tau_val - Algebra::dot(link.S, link.pA);

#ifdef DEBUG
    Algebra::print("m_U", link.U);
    printf("links[%d].d=", i);
    double d1 = Algebra::to_double(link.d);
    printf("%f\n", d1);
    printf("links[%d].u=", i);
    double u = Algebra::to_double(link.u);
    printf("%f\n", u);
#endif

    assert(link.joint_type == JOINT_FIXED || link.d > Algebra::zero());
    Scalar invd = link.joint_type == JOINT_FIXED ? Algebra::zero()
                                                 : Algebra::one() / link.d;
#ifdef DEBUG
    printf("invd[%d]=%f\n", i, Algebra::to_double(invd));
#endif
    auto u_dinv_ut =
        ArticulatedBodyInertia::mul_transpose(link.U * invd, link.U);
    ArticulatedBodyInertia Ia = link.abi - u_dinv_ut;
    ForceVector Ia_c = Ia * link.c;
    ForceVector pa = link.pA + Ia_c + link.U * (link.u * invd);
#ifdef DEBUG
    Algebra::print("u_dinv_ut", u_dinv_ut);
    Algebra::print("Ia", Ia);
    Algebra::print("Ia*c", Ia_c);
    Algebra::print("pa", pa);
#endif

    ForceVector delta_pA = link.X_parent.apply(pa);
#ifdef DEBUG
    Algebra::print("delta_pA", delta_pA);
#endif
    ArticulatedBodyInertia delta_I = link.X_parent.apply(Ia);
    if (parent >= 0) {
      links[parent].pA += delta_pA;
      links[parent].abi += delta_I;
#ifdef DEBUG
      Algebra::print("pa update", links[parent].pA);
      Algebra::print("mIA", links[parent].abi);
#endif
    } else if (is_floating) {
      base_bias_force += delta_pA;
      base_abi += delta_I;
#ifdef DEBUG
      Algebra::print("base_abi", base_abi);
      Algebra::print("base_bias_force", base_bias_force);
      Algebra::print("delta_I", delta_I);
      Algebra::print("delta_pA", delta_pA);
#endif
    }
  }

  if (is_floating) {
    // #ifdef NEURAL_SIM
    //       NEURAL_ASSIGN(base_bias_force[0], "base_bias_force_0");
    //       NEURAL_ASSIGN(base_bias_force[1], "base_bias_force_1");
    //       NEURAL_ASSIGN(base_bias_force[2], "base_bias_force_2");
    //       NEURAL_ASSIGN(base_bias_force[3], "base_bias_force_3");
    //       NEURAL_ASSIGN(base_bias_force[4], "base_bias_force_4");
    //       NEURAL_ASSIGN(base_bias_force[5], "base_bias_force_5");
    // #endif

    base_acceleration = -base_abi.inv_mul(base_bias_force);

  } else {
    base_acceleration = -spatial_gravity;
  }

  for (int i = 0; i < static_cast<int>(links.size()); i++) {
    const Link &link = links[i];
    int parent = link.parent_index;
    const Transform &X_parent = link.X_parent;
    const MotionVector &a_parent =
        (parent >= 0) ? links[parent].a : base_acceleration;
#if DEBUG
    if (parent < 0) {
      printf("final loop for parent %i\n", parent);
      Algebra::print("base_abi", base_abi);
      Algebra::print("base_bias_force", base_bias_force);
      Algebra::print("a_parent", a_parent);
    }
#endif

    MotionVector xpa = X_parent.apply(a_parent);
    link.a = xpa + link.c;
#if DEBUG
    Algebra::print("xpa", xpa);
    Algebra::print("a'", link.a);
#endif
    // model.a[i] = X_parent.apply(model.a[parent]) + model.c[i];
    // LOG << "a'[" << i << "] = " << model.a[i].transpose() << std::endl;

    // if (model.mJoints[i].mDoFCount == 1
    {
      Scalar invd = link.joint_type == JOINT_FIXED ? Algebra::zero()
                                                   : Algebra::one() / link.d;
      Scalar Ut_a = Algebra::dot(link.U, link.a);
      Scalar u_Ut_a = link.u - Ut_a;
      Scalar qdd_val = Algebra::zero();
      if (link.qd_index >= 0) {
        qdd_val = invd * u_Ut_a;
        qdd[link.qd_index] = qdd_val;
      }
      link.a = link.a + link.S * qdd_val;
      Algebra::print("a", link.a);
    }
  }
  if (is_floating) {
    base_acceleration += spatial_gravity;
    for (int i = 0; i < 6; i++) {
      qdd[i] = base_acceleration[i];
    }
  } else {
    base_acceleration.set_zero();
  }
}