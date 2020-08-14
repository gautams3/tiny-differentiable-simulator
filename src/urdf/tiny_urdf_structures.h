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

#include <assert.h>

#include <map>
#include <string>
#include <vector>

#include "../geometry.h"  // for TinyGeometryTypes

template <typename Algebra>
struct TinyUrdfInertial {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  TinyUrdfInertial()
      : mass(Algebra::zero()),
        inertia_xxyyzz(Vector3::zero()),
        origin_rpy(Vector3::zero()),
        origin_xyz(Vector3::zero()) {}
  Scalar mass;
  Vector3 inertia_xxyyzz;
  Vector3 origin_rpy;
  Vector3 origin_xyz;
};

template <typename Algebra>
struct TinyUrdfContact {
  using Scalar = typename Algebra::Scalar;

  TinyUrdfContact()
      : lateral_friction(Algebra::fraction(1, 2)),
        restitution(Algebra::fraction(0, 1)),
        stiffness(Algebra::fraction(1, 1)),
        damping(Algebra::fraction(0, 1)) {}

  Scalar lateral_friction;
  Scalar restitution;
  Scalar stiffness;
  Scalar damping;
};

template <typename Algebra>
struct TinyVisualMaterial {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  Vector3 material_rgb;
  std::string texture_filename;
};

template <typename Algebra>
struct TinyUrdfCollisionSphere {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  TinyUrdfCollisionSphere() : m_radius(Algebra::one()) {}
  Scalar m_radius;
};

template <typename Algebra>
struct TinyUrdfCollisionPlane {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  TinyUrdfCollisionPlane()
      : m_normal(Vector3(Algebra::zero(), Algebra::zero(), Algebra::one())),
        m_constant(Algebra::zero()) {}
  Vector3 m_normal;
  Scalar m_constant;
};

template <typename Algebra>
struct TinyUrdfCollisionCapsule {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  TinyUrdfCollisionCapsule()
      : m_radius(Algebra::one()), m_length(Algebra::one()) {}
  Scalar m_radius;
  Scalar m_length;
};

template <typename Algebra>
struct TinyUrdfCollisionBox {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  TinyUrdfCollisionBox()
      : m_extents(Vector3(Algebra::one(), Algebra::one(), Algebra::one())) {}
  Vector3 m_extents;
};

template <typename Algebra>
struct TinyUrdfCollisionMesh {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  TinyUrdfCollisionMesh()
      : m_scale(Vector3(Algebra::one(), Algebra::one(), Algebra::one())) {}
  std::string m_file_name;
  Vector3 m_scale;
};

template <typename Algebra>
struct TinyUrdfGeometry {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  TinyUrdfGeometry() : geom_type(TINY_MAX_GEOM_TYPE) {}

  // pybind11 doesn't like enum TinyGeometryTypes
  int geom_type;  // see TinyGeometryTypes in tiny_geometry.h

  TinyUrdfCollisionSphere<Algebra> m_sphere;
  TinyUrdfCollisionCapsule<Algebra> m_capsule;
  TinyUrdfCollisionBox<Algebra> m_box;
  TinyUrdfCollisionMesh<Algebra> m_mesh;
  TinyUrdfCollisionPlane<Algebra> m_plane;
};

template <typename Algebra>
struct TinyUrdfVisual {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  typedef ::TinyUrdfGeometry<Algebra> TinyUrdfGeometry;
  typedef ::TinyVisualMaterial<Algebra> TinyVisualMaterial;
  TinyUrdfVisual()
      : origin_xyz(Vector3::zero()),
        origin_rpy(Vector3::zero()),
        has_local_material(false),
        sync_visual_body_uid1(-1),
        sync_visual_body_uid2(-1)

  {}
  Vector3 origin_rpy;
  Vector3 origin_xyz;
  TinyUrdfGeometry geometry;
  std::string material_name;
  TinyVisualMaterial m_material;
  std::string visual_name;
  bool has_local_material;
  int sync_visual_body_uid1;
  int sync_visual_body_uid2;
};

template <typename Algebra>
struct TinyUrdfCollision {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  typedef ::TinyUrdfGeometry<Algebra> TinyUrdfGeometry;

  TinyUrdfCollision()
      : origin_xyz(Vector3::zero()),
        origin_rpy(Vector3::zero()),
        collision_group(0),
        collision_mask(0),
        flags(0) {}
  Vector3 origin_xyz;
  Vector3 origin_rpy;

  std::string collision_name;
  int collision_group;
  int collision_mask;
  int flags;
  TinyUrdfGeometry geometry;
};

template <typename Algebra>
struct TinyUrdfLink {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  typedef ::TinyUrdfCollision<Algebra> TinyUrdfCollision;
  typedef ::TinyUrdfVisual<Algebra> TinyUrdfVisual;
  typedef ::TinyUrdfContact<Algebra> TinyUrdfContact;

  TinyUrdfLink() : m_parent_index(-2) {}
  std::string link_name;
  TinyUrdfInertial<Algebra> urdf_inertial;
  std::vector<TinyUrdfVisual> urdf_visual_shapes;
  std::vector<TinyUrdfCollision> urdf_collision_shapes;
  std::vector<int> child_link_indices;
  TinyUrdfContact contact_info;
  int m_parent_index;
};

template <typename Algebra>
struct TinyUrdfJoint {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  typedef ::TinyUrdfLink<Algebra> TinyUrdfLink;

  TinyUrdfJoint()
      : joint_type(JOINT_INVALID),
        joint_lower_limit(Algebra::one()),
        joint_upper_limit(Algebra::zero()),
        joint_origin_xyz(
            Vector3(Algebra::zero(), Algebra::zero(), Algebra::zero())),
        joint_origin_rpy(
            Vector3(Algebra::zero(), Algebra::zero(), Algebra::zero())),
        joint_axis_xyz(
            Vector3(Algebra::zero(), Algebra::zero(), Algebra::one())) {}
  std::string joint_name;
  // pybind11 doesn't like enum TinyJointType
  int joint_type;
  Scalar joint_lower_limit;
  Scalar joint_upper_limit;
  std::string parent_name;
  std::string child_name;
  Vector3 joint_origin_xyz;
  Vector3 joint_origin_rpy;
  Vector3 joint_axis_xyz;
};

enum TinyConversionReturnCode {
  kCONVERSION_OK = 1,
  kCONVERSION_JOINT_FAILED,
};

template <typename Algebra>
struct TinyUrdfStructures {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  std::string m_robot_name;
  std::vector<TinyUrdfLink<Algebra> > m_base_links;
  std::vector<TinyUrdfLink<Algebra> > m_links;
  std::vector<TinyUrdfJoint<Algebra> > m_joints;
  std::map<std::string, int> m_name_to_link_index;
  std::map<std::string, TinyVisualMaterial<Algebra> > m_materials;
};
