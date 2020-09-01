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

#include <string>
#include <vector>

// #include "tiny_constraint_solver.h"
#include "geometry.hpp"
// #include "tiny_mb_constraint_solver.h"
#include "multi_body.hpp"
// #include "tiny_rigid_body.h"

namespace tds {
template <typename Algebra>
class World {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using Matrix3 = typename Algebra::Matrix3;
  // typedef ::RigidBody<Algebra> RigidBody;
  typedef tds::MultiBody<Algebra> MultiBody;
  typedef tds::Geometry<Algebra> Geometry;
  typedef tds::Transform<Algebra> Transform;

  typedef tds::Capsule<Algebra> Capsule;
  typedef tds::Sphere<Algebra> Sphere;
  typedef tds::Plane<Algebra> Plane;
  // std::vector<RigidBody*> bodies;

  std::vector<MultiBody*> multi_bodies;

  Vector3 gravity_acceleration;

  std::vector<Geometry*> geoms;

  // CollisionDispatcher<Algebra> dispatcher;

 public:
  SubmitProfileTiming profileTimingFunc{nullptr};
  // ConstraintSolver<Algebra>* constraint_solver{nullptr};
  // MultiBodyConstraintSolver<Algebra>*
  //     mb_constraint_solver{nullptr};

  int num_solver_iterations{50};

  // contact settings
  Scalar default_friction{Algebra::fraction(2, 10)};
  Scalar default_restitution{Algebra::zero()};

  explicit World(Scalar gravity_z = Algebra::fraction(-981, 100))
      : gravity_acceleration(Algebra::zero(), Algebra::zero(), gravity_z) {}
  // constraint_solver(
  //     new ConstraintSolver<Algebra>),
  // mb_constraint_solver(
  //     new MultiBodyConstraintSolver<Algebra>) {}

  inline void submitProfileTiming(const std::string& name) {
    // if (m_profileTimingFunc) {
    //   profileTimingFunc(name);
    // }
  }
  virtual ~World() { clear(); }

  void clear() {
    for (int i = 0; i < geoms.size(); i++) {
      delete geoms[i];
    }
    geoms.clear();

    // for (int i = 0; i < bodies.size(); i++) {
    //   delete bodies[i];
    // }
    // bodies.clear();

    for (int i = 0; i < multi_bodies.size(); i++) {
      delete multi_bodies[i];
    }
    multi_bodies.clear();

    // if (m_constraint_solver) {
    //   delete constraint_solver;
    //   constraint_solver = nullptr;
    // }
  }

  const Vector3& get_gravity() const { return gravity_acceleration; }

  void set_gravity(const Vector3& gravity) { gravity_acceleration = gravity; }

  // ConstraintSolver<Algebra>* get_constraint_solver() {
  //   return constraint_solver;
  // }

  Capsule* create_capsule(Scalar radius, Scalar length) {
    Capsule* capsule = new Capsule(radius, length);
    geoms.push_back(capsule);
    return capsule;
  }

  Plane* create_plane() {
    Plane* plane = new Plane();
    geoms.push_back(plane);
    return plane;
  }

  Sphere* create_sphere(Scalar radius) {
    Sphere* sphere = new Sphere(radius);
    geoms.push_back(sphere);
    return sphere;
  }

  // CollisionDispatcher<Algebra>
  // get_collision_dispatcher() {
  //   return dispatcher;
  // }

  // RigidBody* create_rigid_body(Scalar mass, const Geometry* geom) {
  //   RigidBody* body = new RigidBody(mass, geom);
  //   this->m_bodies.push_back(body);
  //   return body;
  // }

  MultiBody* create_multi_body() {
    MultiBody* body = new MultiBody();
    multi_bodies.push_back(body);
    return body;
  }

  // std::vector<ContactPointRigidBody<Algebra>>
  //     allContacts;
  // std::vector<std::vector<ContactPointMultiBody<Algebra>>>
  //     allMultiBodyContacts;
  // std::vector<std::vector<ContactPointMultiBody<Algebra>>>
  //     additional_MultiBodyContacts;

  // std::vector<ContactPoint<Algebra>> contacts;

  // static void compute_contacts_rigid_body_internal(
  //     std::vector<RigidBody*> bodies,
  //     CollisionDispatcher<Algebra>* dispatcher,
  //     std::vector<ContactPointRigidBody<Algebra>>&
  //         contactsOut,
  //     const Scalar& restitution, const Scalar& friction) {
  //   std::vector<ContactPoint<Algebra>> contacts;
  //   {
  //     for (int i = 0; i < bodies.size(); i++) {
  //       for (int j = i + 1; j < bodies.size(); j++) {
  //         contacts.reserve(1);
  //         contacts.resize(0);

  //         int numContacts = dispatcher->computeContacts(
  //             bodies[i]->m_geometry, bodies[i]->m_world_pose,
  //             bodies[j]->m_geometry, bodies[j]->m_world_pose, contacts);
  //         for (int c = 0; c < numContacts; c++) {
  //           ContactPointRigidBody<Algebra> rb_pt;
  //           ContactPoint<Algebra>& pt = rb_pt;
  //           pt = contacts[c];
  //           rb_pt.m_rigid_body_a = bodies[i];
  //           rb_pt.m_rigid_body_b = bodies[j];
  //           // TODO(erwincoumans): combine friction and restitution based on
  //           // material properties of the two touching bodies
  //           rb_pt.m_restitution = restitution;
  //           rb_pt.m_friction = friction;
  //           contactsOut.push_back(rb_pt);
  //         }
  //       }
  //     }
  //   }
  // }

  // std::vector<ContactPointRigidBody<Algebra>>
  // compute_contacts_rigid_body(
  //     std::vector<RigidBody*> bodies,
  //     CollisionDispatcher<Algebra>* dispatcher) {
  //   std::vector<ContactPointRigidBody<Algebra>>
  //       contactsOut;
  //   compute_contacts_rigid_body_internal(bodies, dispatcher, contactsOut,
  //                                        default_restitution,
  //                                        default_friction);
  //   return contactsOut;
  // }

  // static void compute_contacts_multi_body_internal(
  //     std::vector<MultiBody*> multi_bodies,
  //     CollisionDispatcher<Algebra>* dispatcher,
  //     std::vector<
  //         std::vector<ContactPointMultiBody<Algebra>>>&
  //         contacts_out,
  //     const Scalar& restitution, const Scalar& friction) {
  //   int num_multi_bodies = multi_bodies.size();
  //   for (int i = 0; i < num_multi_bodies; i++) {
  //     MultiBody* mb_a = multi_bodies[i];
  //     int num_links_a = mb_a->m_links.size();
  //     for (int j = i + 1; j < multi_bodies.size(); j++) {
  //       std::vector<ContactPoint<Algebra>> contacts;

  //       MultiBody* mb_b = multi_bodies[j];
  //       int num_links_b = mb_b->m_links.size();
  //       std::vector<ContactPointMultiBody<Algebra>>
  //           contacts_ab;
  //       for (int ii = -1; ii < num_links_a; ii++) {
  //         const Transform& world_transform_a =
  //             mb_a->get_world_transform(ii);

  //         int num_geoms_a = mb_a->get_collision_geometries(ii).size();
  //         for (int iii = 0; iii < num_geoms_a; iii++) {
  //           const Geometry* geom_a =
  //               mb_a->get_collision_geometries(ii)[iii];
  //           Pose<Algebra> pose_a;
  //           const Transform& local_a =
  //               mb_a->get_collision_transforms(ii)[iii];
  //           Transform tr_a = world_transform_a * local_a;
  //           pose_a.m_position = tr_a.m_translation;
  //           tr_a.m_rotation.getRotation(pose_a.m_orientation);

  //           for (int jj = -1; jj < num_links_b; jj++) {
  //             const Transform& world_transform_b =
  //                 mb_b->get_world_transform(jj);
  //             int num_geoms_b = mb_b->get_collision_geometries(jj).size();
  //             for (int jjj = 0; jjj < num_geoms_b; jjj++) {
  //               const Geometry* geom_b =
  //                   mb_b->get_collision_geometries(jj)[jjj];
  //               Pose<Algebra> pose_b;
  //               const Transform& local_b =
  //                   mb_b->get_collision_transforms(jj)[jjj];
  //               Transform tr_b = world_transform_b * local_b;
  //               pose_b.m_position = tr_b.m_translation;
  //               tr_b.m_rotation.getRotation(pose_b.m_orientation);

  //               // printf("\tworld_transform_b: %.3f  %.3f  %.3f\n",
  //               // world_transform_b.m_translation[0],
  //               // world_transform_b.m_translation[1],
  //               // world_transform_b.m_translation[2]);
  //               //                printf("\tpose_b: %.3f  %.3f  %.3f\n",
  //               //                pose_b.m_position[0], pose_b.m_position[1],
  //               //                pose_b.m_position[2]);
  //               contacts.reserve(1);
  //               contacts.resize(0);
  //               int numContacts = dispatcher->computeContacts(
  //                   geom_a, pose_a, geom_b, pose_b, contacts);
  //               for (int c = 0; c < numContacts; c++) {
  //                 ContactPointMultiBody<Algebra> mb_pt;
  //                 ContactPoint<Algebra>& pt = mb_pt;
  //                 pt = contacts[c];
  //                 mb_pt.m_multi_body_a = multi_bodies[i];
  //                 mb_pt.m_multi_body_b = multi_bodies[j];
  //                 mb_pt.m_link_a = ii;
  //                 mb_pt.m_link_b = jj;
  //                 // TODO(erwincoumans): combine friction and restitution
  //                 // based on material properties of the two touching bodies
  //                 mb_pt.m_restitution = restitution;
  //                 mb_pt.m_friction = friction;
  //                 contacts_ab.push_back(mb_pt);
  //               }
  //             }
  //           }
  //           //            printf("\n");
  //           //            fflush(stdout);
  //         }
  //       }

  //       contacts_out.push_back(contacts_ab);
  //     }
  //   }
  // }

  // std::vector<std::vector<ContactPointMultiBody<Algebra>>>
  // compute_contacts_multi_body(
  //     std::vector<MultiBody*> bodies,
  //     CollisionDispatcher<Algebra>* dispatcher) {
  //   std::vector<
  //       std::vector<ContactPointMultiBody<Algebra>>>
  //       contactsOut;
  //   compute_contacts_multi_body_internal(bodies, dispatcher, contactsOut,
  //                                        default_restitution,
  //                                        default_friction);
  //   return contactsOut;
  // }

  // void step(Scalar dt) {
  //   {
  //     allContacts.reserve(1024);
  //     allContacts.resize(0);
  //     allMultiBodyContacts.reserve(1024);
  //     allMultiBodyContacts.resize(0);
  //     submitProfileTiming("apply forces");
  //     for (int i = 0; i < bodies.size(); i++) {
  //       RigidBody* b = bodies[i];
  //       b->apply_gravity(m_gravity_acceleration);
  //       b->apply_force_impulse(dt);
  //       b->clear_forces();
  //     }
  //     submitProfileTiming("");
  //   }
  //   {
  //     submitProfileTiming("compute contacts");
  //     compute_contacts_rigid_body_internal(m_bodies, &m_dispatcher,
  //                                          allContacts, default_restitution,
  //                                          default_friction);
  //     submitProfileTiming("");
  //   }
  //   {
  //     submitProfileTiming("compute multi body contacts");
  //     compute_contacts_multi_body_internal(
  //         multi_bodies, &m_dispatcher, allMultiBodyContacts,
  //         default_restitution, default_friction);
  //     submitProfileTiming("");
  //     allMultiBodyContacts.insert(m_allMultiBodyContacts.end(),
  //                                   additional_MultiBodyContacts.begin(),
  //                                   additional_MultiBodyContacts.end());
  //   }

  //   {
  //     submitProfileTiming("solve constraints");
  //     for (int i = 0; i < num_solver_iterations; i++) {
  //       for (int c = 0; c < allContacts.size(); c++) {
  //         constraint_solver->resolveCollision(m_allContacts[c], dt);
  //       }
  //     }
  //     // use outer loop in case the multi-body constraint solver requires it
  //     // (e.g. sequential impulse method)
  //     int mb_solver_iters;
  //     if (!m_mb_constraint_solver->needs_outer_iterations) {
  //       mb_solver_iters = 1;
  //     } else {
  //       mb_solver_iters = num_solver_iterations;
  //     }
  //     // std::cout << "Resolving " << allMultiBodyContacts.size()
  //     //           << " contacts.\n";
  //     for (int i = 0; i < mb_solver_iters; i++) {
  //       for (int c = 0; c < allMultiBodyContacts.size(); c++) {
  //         mb_constraint_solver->resolveCollision(m_allMultiBodyContacts[c],
  //                                                  dt);
  //       }
  //     }
  //     submitProfileTiming("");
  //   }

  //   {
  //     submitProfileTiming("integrate");
  //     for (int i = 0; i < bodies.size(); i++) {
  //       RigidBody* b = bodies[i];
  //       b->integrate(dt);
  //     }
  //     submitProfileTiming("");
  //   }
  // }
};
}  // namespace tds
