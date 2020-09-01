/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "contact_point.hpp"
#include "rigid_body.hpp"

namespace tds {
template <typename Algebra>
struct RigidBodyContactPoint : public ContactPoint<Algebra> {
  typedef tds::RigidBody<Algebra> RigidBody;
  using Scalar = typename Algebra::Scalar;
  RigidBody* rigid_body_a{nullptr};
  RigidBody* rigid_body_b{nullptr};
  Scalar restitution;
  Scalar friction;
};

template <typename Algebra>
class RigidBodyConstraintSolver {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

 public:
  virtual ~RigidBodyConstraintSolver() = default;

  // Note that the LCP(A, b) is not explicitly constructed.
  // Baumgarte stabilization is used to reduce positional drift.See description
  // in Michael Cline's thesis:
  // https ://www.cs.ubc.ca/grads/resources/thesis/Nov02/Michael_Cline.pdf

  // Args:
  // cp: contact point
  // dt : delta time (in seconds)
  virtual void resolve_collision(RigidBodyContactPoint<Algebra>& cp,
                                const Scalar& dt) const {
    Scalar erp =
        Algebra::fraction(1, 10);  // BAUMGARTE_ERROR_REDUCTION_PARAMETER
    const Vector3& world_point_a = cp.world_point_on_a;
    const Vector3& world_point_b = cp.world_point_on_b;
    Vector3 rel_pos_a = world_point_a - cp.rigid_body_a->world_pose().position;
    Vector3 rel_pos_b = world_point_b - cp.rigid_body_b->world_pose().position;
    Scalar baumgarte_rel_vel = Algebra::zero();

    if (cp.distance < Algebra::zero()) {
      baumgarte_rel_vel = erp * cp.distance / dt;
      Vector3 vel_a = cp.rigid_body_a->get_velocity(rel_pos_a);
      Vector3 vel_b = cp.rigid_body_b->get_velocity(rel_pos_b);
      Vector3 rel_vel = vel_a - vel_b;
      Scalar normal_rel_vel = Algebra::dot(cp.world_normal_on_b, rel_vel);
      if (normal_rel_vel < Algebra::zero()) {
        Vector3 temp1 =
            Algebra::dot(cp.rigid_body_a->inv_inertia_world(),
                         Algebra::cross(rel_pos_a, cp.world_normal_on_b));
        Vector3 temp2 =
            Algebra::dot(cp.rigid_body_b->inv_inertia_world(),
                         Algebra::cross(rel_pos_b, cp.world_normal_on_b));
        Scalar ang = Algebra::dot(cp.world_normal_on_b,
                                  Algebra::cross(temp1, rel_pos_a) +
                                      Algebra::cross(temp2, rel_pos_b));
        Scalar impulse =
            (-(Algebra::one() + cp.restitution) * normal_rel_vel -
             baumgarte_rel_vel) /
            (cp.rigid_body_a->inv_mass() + cp.rigid_body_b->inv_mass() + ang);
        if (impulse > Algebra::zero()) {
          Vector3 impulse_vector = impulse * cp.world_normal_on_b;
          cp.rigid_body_a->apply_impulse(impulse_vector, rel_pos_a);
          cp.rigid_body_b->apply_impulse(-impulse_vector, rel_pos_b);

          Vector3 lateral_rel_vel =
              rel_vel - normal_rel_vel * cp.world_normal_on_b;
          Scalar friction_impulse_trial =
              (lateral_rel_vel.length()) /
              (cp.rigid_body_a->inv_mass() + cp.rigid_body_b->inv_mass() + ang);

          Scalar friction_coeffcient = cp.friction;
          Scalar friction_impulse;
          if (friction_impulse_trial < friction_coeffcient * impulse) {
            friction_impulse = friction_impulse_trial;
          } else {
            friction_impulse = friction_coeffcient * impulse;
          }

          if ((lateral_rel_vel.length()) > Algebra::fraction(1, 10000)) {
            Vector3 friction_dir =
                lateral_rel_vel * (Algebra::one() / lateral_rel_vel.length());
            cp.rigid_body_a->apply_impulse(-friction_impulse * friction_dir,
                                           rel_pos_a);
            cp.rigid_body_b->apply_impulse(friction_impulse * friction_dir,
                                           rel_pos_b);
          }
        }
      }
    }
  }
};
}  // namespace tds
