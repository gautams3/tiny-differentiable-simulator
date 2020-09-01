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

#include <vector>

#include "math/pose.hpp"

namespace tds {
enum GeometryTypes {
  TINY_SPHERE_TYPE = 0,
  TINY_PLANE_TYPE,
  TINY_CAPSULE_TYPE,
  TINY_MESH_TYPE,      // only for visual shapes at the moment
  TINY_BOX_TYPE,       // only for visual shapes at the moment
  TINY_CYLINDER_TYPE,  // unsupported
  TINY_MAX_GEOM_TYPE,
};

template <typename Algebra>
class Geometry {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  int type;

 public:
  explicit Geometry(int type) : type(type) {}
  virtual ~Geometry() = default;
  int get_type() const { return type; }
};

template <typename Algebra>
class Sphere : public Geometry<Algebra> {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  Scalar radius;

 public:
  explicit Sphere(Scalar radius)
      : Geometry<Algebra>(TINY_SPHERE_TYPE), radius(radius) {}

  Scalar get_radius() const { return radius; }

  Vector3 compute_local_inertia(Scalar mass) const {
    Scalar elem = Algebra::fraction(4, 10) * mass * radius * radius;
    return Vector3(elem, elem, elem);
  }
};

// capsule aligned with the Z axis
template <typename Algebra>
class Capsule : public Geometry<Algebra> {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  Scalar radius;
  Scalar length;

 public:
  explicit Capsule(Scalar radius, Scalar length)
      : Geometry<Algebra>(TINY_CAPSULE_TYPE), radius(radius), length(length) {}

  Scalar get_radius() const { return radius; }
  Scalar get_length() const { return length; }

  Vector3 compute_local_inertia(Scalar mass) const {
    Scalar lx = Algebra::fraction(2, 1) * (radius);
    Scalar ly = Algebra::fraction(2, 1) * (radius);
    Scalar lz = length + Algebra::fraction(2, 1) * (radius);
    Scalar x2 = lx * lx;
    Scalar y2 = ly * ly;
    Scalar z2 = lz * lz;
    Scalar scaledmass = mass * Algebra::fraction(1, 12);

    Vector3 inertia;
    inertia[0] = scaledmass * (y2 + z2);
    inertia[1] = scaledmass * (x2 + z2);
    inertia[2] = scaledmass * (x2 + y2);
    return inertia;
  }
};

template <typename Algebra>
class Plane : public Geometry<Algebra> {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  Vector3 normal;
  Scalar constant;

 public:
  Plane()
      : Geometry<Algebra>(TINY_PLANE_TYPE),
        normal(Algebra::zero(), Algebra::zero(), Algebra::one()),
        constant(Algebra::zero()) {}

  const Vector3& get_normal() const { return normal; }
  Scalar get_constant() const { return constant; }
};

template <typename Algebra>
struct ContactPoint {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  Vector3 world_normal_on_b;
  Vector3 world_point_on_a;
  Vector3 world_point_on_b;
  Scalar distance;
};

template <typename Algebra>
int contactSphereSphere(const Geometry<Algebra>* geomA,
                        const Pose<Algebra>& poseA,
                        const Geometry<Algebra>* geomB,
                        const Pose<Algebra>& poseB,
                        std::vector<ContactPoint<Algebra> >& contactsOut) {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  Scalar CONTACT_EPSILON = Algebra::fraction(1, 100000);

  typedef tds::Sphere<Algebra> Sphere;
  typedef tds::ContactPoint<Algebra> ContactPoint;
  assert(geomA->get_type() == TINY_SPHERE_TYPE);
  assert(geomB->get_type() == TINY_SPHERE_TYPE);
  Sphere* sphereA = (Sphere*)geomA;
  Sphere* sphereB = (Sphere*)geomB;

  Vector3 diff = poseA.m_position - poseB.m_position;
  Scalar length = diff.length();
  Scalar distance = length - (sphereA->get_radius() + sphereB->get_radius());
  Vector3 normal_on_b;
  normal_on_b.setValue(Algebra::one(), Algebra::zero(), Algebra::zero());
  if (length > CONTACT_EPSILON) {
    Vector3 normal_on_b = Algebra::one() / length * diff;
    Vector3 point_a_world =
        poseA.m_position - sphereA->get_radius() * normal_on_b;
    Vector3 point_b_world = point_a_world - distance * normal_on_b;
    ContactPoint pt;
    pt.m_world_normal_on_b = normal_on_b;
    pt.m_world_point_on_a = point_a_world;
    pt.m_world_point_on_b = point_b_world;
    pt.m_distance = distance;
    contactsOut.push_back(pt);
    return 1;
  }
  return 0;
}

template <typename Algebra>
int contactPlaneSphere(const Geometry<Algebra>* geomA,
                       const Pose<Algebra>& poseA,
                       const Geometry<Algebra>* geomB,
                       const Pose<Algebra>& poseB,
                       std::vector<ContactPoint<Algebra> >& contactsOut) {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  typedef tds::Sphere<Algebra> Sphere;
  typedef tds::Plane<Algebra> Plane;
  typedef tds::ContactPoint<Algebra> ContactPoint;
  assert(geomA->get_type() == TINY_PLANE_TYPE);
  assert(geomB->get_type() == TINY_SPHERE_TYPE);
  Plane* planeA = (Plane*)geomA;
  Sphere* sphereB = (Sphere*)geomB;

  Scalar t =
      -(poseB.m_position.dot(-planeA->get_normal()) + planeA->get_constant());
  Vector3 pointAWorld = poseB.m_position + t * -planeA->get_normal();
  Scalar distance = t - sphereB->get_radius();
  Vector3 pointBWorld =
      poseB.m_position - sphereB->get_radius() * planeA->get_normal();
  ContactPoint pt;
  pt.m_world_normal_on_b = -planeA->get_normal();
  pt.m_world_point_on_a = pointAWorld;
  pt.m_world_point_on_b = pointBWorld;
  pt.m_distance = distance;
  contactsOut.push_back(pt);
  return 1;
}

template <typename Algebra>
int contactPlaneCapsule(const Geometry<Algebra>* geomA,
                        const Pose<Algebra>& poseA,
                        const Geometry<Algebra>* geomB,
                        const Pose<Algebra>& poseB,
                        std::vector<ContactPoint<Algebra> >& contactsOut) {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  typedef tds::Pose<Algebra> Pose;
  typedef tds::Plane<Algebra> Plane;
  typedef tds::Capsule<Algebra> Capsule;
  typedef tds::ContactPoint<Algebra> ContactPoint;
  typedef tds::Sphere<Algebra> Sphere;
  assert(geomA->get_type() == TINY_PLANE_TYPE);
  assert(geomB->get_type() == TINY_CAPSULE_TYPE);
  Capsule* capsule = (Capsule*)geomB;

  // create twice a plane-sphere contact
  Sphere sphere(capsule->get_radius());
  // shift the sphere to each end-point
  Pose offset;
  offset.m_orientation.set_identity();
  offset.m_position.setValue(Algebra::zero(), Algebra::zero(),
                             Algebra::fraction(1, 2) * capsule->get_length());
  Pose poseEndSphere = poseB * offset;
  contactPlaneSphere<Algebra>(geomA, poseA, &sphere, poseEndSphere,
                              contactsOut);
  offset.m_position.setValue(Algebra::zero(), Algebra::zero(),
                             Algebra::fraction(-1, 2) * capsule->get_length());
  poseEndSphere = poseB * offset;
  contactPlaneSphere<Algebra>(geomA, poseA, &sphere, poseEndSphere,
                              contactsOut);

  return 2;
}

template <typename Algebra>
struct CollisionDispatcher {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  typedef tds::Geometry<Algebra> Geometry;
  typedef tds::Pose<Algebra> Pose;
  typedef tds::ContactPoint<Algebra> ContactPoint;

  typedef int (*contact_func)(const Geometry* geomA, const Pose& poseA,
                              const Geometry* geomB, const Pose& poseB,
                              std::vector<ContactPoint>& contactsOut);

  contact_func contactFuncs[TINY_MAX_GEOM_TYPE][TINY_MAX_GEOM_TYPE];

  CollisionDispatcher() {
    for (int i = 0; i < TINY_MAX_GEOM_TYPE; i++) {
      for (int j = 0; j < TINY_MAX_GEOM_TYPE; j++) {
        contactFuncs[i][j] = 0;
      }
    }
    contactFuncs[TINY_SPHERE_TYPE][TINY_SPHERE_TYPE] = contactSphereSphere;
    contactFuncs[TINY_PLANE_TYPE][TINY_SPHERE_TYPE] = contactPlaneSphere;
    contactFuncs[TINY_PLANE_TYPE][TINY_CAPSULE_TYPE] = contactPlaneCapsule;
  }

  int computeContacts(const Geometry* geomA, const Pose& poseA,
                      const Geometry* geomB, const Pose& poseB,
                      std::vector<ContactPoint>& contactsOut) {
    contact_func f = contactFuncs[geomA->get_type()][geomB->get_type()];
    if (f) {
      return f(geomA, poseA, geomB, poseB, contactsOut);
    }
    return 0;
  }

  std::vector<ContactPoint> compute_contacts(const Geometry* geomA,
                                             const Pose& poseA,
                                             const Geometry* geomB,
                                             const Pose& poseB) {
    std::vector<ContactPoint> pts;
    int num = computeContacts(geomA, poseA, geomB, poseB, pts);
    return pts;
  }
};
}