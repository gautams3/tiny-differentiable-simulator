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

#ifndef TINY_GEOMETRY_H
#define TINY_GEOMETRY_H

#include <vector>

#include "tiny_pose.h"

enum TinyGeometryTypes {
  TINY_SPHERE_TYPE = 0,
  TINY_PLANE_TYPE,
  TINY_CAPSULE_TYPE,
  TINY_MESH_TYPE,      // only for visual shapes at the moment
  TINY_BOX_TYPE,       // only for visual shapes at the moment
  TINY_CYLINDER_TYPE,  // unsupported
  TINY_MAX_GEOM_TYPE,
};

template <typename Algebra>
class TinyGeometry {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  int m_type;

 public:
  explicit TinyGeometry(int type) : m_type(type) {}
  virtual ~TinyGeometry() {}
  int get_type() const { return m_type; }
};

template <typename Algebra>
class TinySphere : public TinyGeometry<Algebra> {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  Scalar m_radius;

 public:
  explicit TinySphere(Scalar radius)
      : TinyGeometry<Algebra>(TINY_SPHERE_TYPE), m_radius(radius) {}

  Scalar get_radius() const { return m_radius; }

  Vector3 compute_local_inertia(Scalar mass) const {
    Scalar elem = Algebra::fraction(4, 10) * mass * m_radius * m_radius;
    return Vector3(elem, elem, elem);
  }
};

// capsule aligned with the Z axis
template <typename Algebra>
class TinyCapsule : public TinyGeometry<Algebra> {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  Scalar m_radius;
  Scalar m_length;

 public:
  explicit TinyCapsule(Scalar radius, Scalar length)
      : TinyGeometry<Algebra>(TINY_CAPSULE_TYPE),
        m_radius(radius),
        m_length(length) {}

  Scalar get_radius() const { return m_radius; }
  Scalar get_length() const { return m_length; }

  Vector3 compute_local_inertia(Scalar mass) const {
    Scalar lx = Algebra::fraction(2, 1) * (m_radius);
    Scalar ly = Algebra::fraction(2, 1) * (m_radius);
    Scalar lz = m_length + Algebra::fraction(2, 1) * (m_radius);
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
class TinyPlane : public TinyGeometry<Algebra> {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  Vector3 m_normal;
  Scalar m_constant;

 public:
  TinyPlane()
      : TinyGeometry<Algebra>(TINY_PLANE_TYPE),
        m_normal(Algebra::zero(), Algebra::zero(), Algebra::one()),
        m_constant(Algebra::zero()) {}

  const Vector3& get_normal() const { return m_normal; }
  Scalar get_constant() const { return m_constant; }
};

template <typename Algebra>
struct TinyContactPoint {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  Vector3 m_world_normal_on_b;
  Vector3 m_world_point_on_a;
  Vector3 m_world_point_on_b;
  Scalar m_distance;
};

template <typename Algebra>
int contactSphereSphere(

    using Scalar = typename Algebra::Scalar;
    using Vector3 = typename Algebra::Vector3;

    const TinyGeometry<Algebra>* geomA, const TinyPose<Algebra>& poseA,
    const TinyGeometry<Algebra>* geomB, const TinyPose<Algebra>& poseB,
    std::vector<TinyContactPoint<Algebra> >& contactsOut) {
  Scalar CONTACT_EPSILON = Algebra::fraction(1, 100000);

  typedef ::TinySphere<Algebra> TinySphere;
  typedef ::TinyContactPoint<Algebra> TinyContactPoint;
  assert(geomA->get_type() == TINY_SPHERE_TYPE);
  assert(geomB->get_type() == TINY_SPHERE_TYPE);
  TinySphere* sphereA = (TinySphere*)geomA;
  TinySphere* sphereB = (TinySphere*)geomB;

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
    TinyContactPoint pt;
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
int contactPlaneSphere(
    const TinyGeometry<Algebra>* geomA, const TinyPose<Algebra>& poseA,
    const TinyGeometry<Algebra>* geomB, const TinyPose<Algebra>& poseB,
    std::vector<TinyContactPoint<Algebra> >& contactsOut) {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  typedef ::TinySphere<Algebra> TinySphere;
  typedef ::TinyPlane<Algebra> TinyPlane;
  typedef ::TinyContactPoint<Algebra> TinyContactPoint;
  assert(geomA->get_type() == TINY_PLANE_TYPE);
  assert(geomB->get_type() == TINY_SPHERE_TYPE);
  TinyPlane* planeA = (TinyPlane*)geomA;
  TinySphere* sphereB = (TinySphere*)geomB;

  Scalar t =
      -(poseB.m_position.dot(-planeA->get_normal()) + planeA->get_constant());
  Vector3 pointAWorld = poseB.m_position + t * -planeA->get_normal();
  Scalar distance = t - sphereB->get_radius();
  Vector3 pointBWorld =
      poseB.m_position - sphereB->get_radius() * planeA->get_normal();
  TinyContactPoint pt;
  pt.m_world_normal_on_b = -planeA->get_normal();
  pt.m_world_point_on_a = pointAWorld;
  pt.m_world_point_on_b = pointBWorld;
  pt.m_distance = distance;
  contactsOut.push_back(pt);
  return 1;
}

template <typename Algebra>
int contactPlaneCapsule(

    using Scalar = typename Algebra::Scalar;
    using Vector3 = typename Algebra::Vector3;

    const TinyGeometry<Algebra>* geomA, const TinyPose<Algebra>& poseA,
    const TinyGeometry<Algebra>* geomB, const TinyPose<Algebra>& poseB,
    std::vector<TinyContactPoint<Algebra> >& contactsOut) {
  typedef ::TinyPose<Algebra> TinyPose;
  typedef ::TinyPlane<Algebra> TinyPlane;
  typedef ::TinyCapsule<Algebra> TinyCapsule;
  typedef ::TinyContactPoint<Algebra> TinyContactPoint;
  typedef ::TinySphere<Algebra> TinySphere;
  assert(geomA->get_type() == TINY_PLANE_TYPE);
  assert(geomB->get_type() == TINY_CAPSULE_TYPE);
  TinyCapsule* capsule = (TinyCapsule*)geomB;

  // create twice a plane-sphere contact
  TinySphere sphere(capsule->get_radius());
  // shift the sphere to each end-point
  TinyPose offset;
  offset.m_orientation.set_identity();
  offset.m_position.setValue(Algebra::zero(), Algebra::zero(),
                             Algebra::fraction(1, 2) * capsule->get_length());
  TinyPose poseEndSphere = poseB * offset;
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
struct TinyCollisionDispatcher {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  typedef ::TinyGeometry<Algebra> TinyGeometry;
  typedef ::TinyPose<Algebra> TinyPose;
  typedef ::TinyContactPoint<Algebra> TinyContactPoint;

  typedef int (*contact_func)(const TinyGeometry* geomA, const TinyPose& poseA,
                              const TinyGeometry* geomB, const TinyPose& poseB,
                              std::vector<TinyContactPoint>& contactsOut);

  contact_func m_contactFuncs[TINY_MAX_GEOM_TYPE][TINY_MAX_GEOM_TYPE];

  TinyCollisionDispatcher() {
    for (int i = 0; i < TINY_MAX_GEOM_TYPE; i++) {
      for (int j = 0; j < TINY_MAX_GEOM_TYPE; j++) {
        m_contactFuncs[i][j] = 0;
      }
    }
    m_contactFuncs[TINY_SPHERE_TYPE][TINY_SPHERE_TYPE] = contactSphereSphere;
    m_contactFuncs[TINY_PLANE_TYPE][TINY_SPHERE_TYPE] = contactPlaneSphere;
    m_contactFuncs[TINY_PLANE_TYPE][TINY_CAPSULE_TYPE] = contactPlaneCapsule;
  }

  int computeContacts(const TinyGeometry* geomA, const TinyPose& poseA,
                      const TinyGeometry* geomB, const TinyPose& poseB,
                      std::vector<TinyContactPoint>& contactsOut) {
    contact_func f = m_contactFuncs[geomA->get_type()][geomB->get_type()];
    if (f) {
      return f(geomA, poseA, geomB, poseB, contactsOut);
    }
    return 0;
  }

  std::vector<TinyContactPoint> compute_contacts(const TinyGeometry* geomA,
                                                 const TinyPose& poseA,
                                                 const TinyGeometry* geomB,
                                                 const TinyPose& poseB) {
    std::vector<TinyContactPoint> pts;
    int num = computeContacts(geomA, poseA, geomB, poseB, pts);
    return pts;
  }
};

#endif  // TINY_GEOMETRY_H
