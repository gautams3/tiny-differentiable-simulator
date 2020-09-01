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

// #include "pybullet_urdf_import.h"
// #include "tiny_actuator.h"
// #include "tiny_double_utils.h"
#include "../multi_body.hpp"
#include "tiny_urdf_parser.h"
#include "urdf_to_multi_body.hpp"
#include "world.hpp"

namespace tds {
template <typename Algebra>
struct UrdfCache {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  typedef tds::UrdfStructures<Algebra> UrdfStructures;
  // typedef ::PyBulletUrdfImport<Algebra> UrdfImport;
  // typedef b3RobotSimulatorLoadUrdfFileArgs UrdfFileArgs;

  std::map<std::string, UrdfStructures> data;

  // template <typename VisualizerAPI>
  // const UrdfStructures& retrieve(const std::string& urdf_filename,
  //                                VisualizerAPI* sim, VisualizerAPI* vis,
  //                                UrdfFileArgs args = UrdfFileArgs(),
  //                                bool ignore_cache = false) {
  //   if (ignore_cache || data.find(urdf_filename) == data.end()) {
  //     printf("Loading URDF \"%s\".\n", urdf_filename.c_str());
  //     int robotId = sim->loadURDF(urdf_filename, args);
  //     if (robotId < 0) {
  //       std::cerr << "Error: Could not load URDF file \"" << urdf_filename
  //                 << "\".\n";
  //       exit(1);
  //     }
  //     data[urdf_filename] = UrdfStructures();
  //     UrdfImport::extract_urdf_structs(data[urdf_filename], robotId, *sim,
  //                                      *vis);
  //     sim->removeBody(robotId);
  //   }
  //   return data[urdf_filename];
  // }

  const UrdfStructures& retrieve(const std::string& urdf_filename,
                                 bool ignore_cache = false) {
    if (ignore_cache || data.find(urdf_filename) == data.end()) {
      printf("Loading URDF \"%s\".\n", urdf_filename.c_str());
      UrdfParser<Algebra> parser;
      data[urdf_filename] = parser.load_urdf(urdf_filename);
    }
    return data[urdf_filename];
  }

  MultiBody<Algebra>* construct(const std::string& urdf_filename,
                                tds::World<Algebra>& world,
                                bool ignore_cache = false,
                                bool is_floating = false) {
    MultiBody<Algebra>* mb = world.create_multi_body();
    const auto& urdf_data = retrieve(urdf_filename, ignore_cache);
    UrdfToMultiBody<Algebra>::convert_to_multi_body(urdf_data, world, *mb);
    mb->set_floating_base(is_floating);
    mb->initialize();
    return mb;
  }

  // template <typename VisualizerAPI>
  // MultiBody<Algebra>* construct(const std::string& urdf_filename,
  //                                         World<Algebra>& world,
  //                                         VisualizerAPI* sim,
  //                                         VisualizerAPI* vis,
  //                                         bool ignore_cache = false,
  //                                         bool is_floating = false) {
  //   b3RobotSimulatorLoadUrdfFileArgs args;
  //   args.m_flags |= URDF_MERGE_FIXED_LINKS;
  //   MultiBody<Algebra>* mb = world.create_multi_body();
  //   const auto& urdf_data =
  //       retrieve(urdf_filename, sim, vis, args, ignore_cache);
  //   UrdfToMultiBody<Algebra>::convert_to_multi_body(urdf_data, world,
  //                                                             *mb);
  //   mb->isFloating = is_floating;
  //   mb->initialize();
  //   return mb;
  // }
};

/**
 * Provides the system construction function to System and derived classes.
 */
// template <template <typename> typename Actuator = Actuator>
// struct SystemConstructor {
//   std::string system_urdf_filename;
//   // if empty, no ground plane is used
//   std::string plane_urdf_filename{""};

//   bool is_floating{false};

//   // settings for stiffness and damping of all joints in the system
//   double joint_stiffness{0};
//   double joint_damping{0};

//   std::vector<int> control_indices;

//   // Actuator<Algebra>* actuator{nullptr};

//   explicit SystemConstructor(const std::string& system_urdf_filename,
//                                  const std::string& plane_urdf_filename = "")
//       : system_urdf_filename(system_urdf_filename),
//         plane_urdf_filename(plane_urdf_filename) {}

//   SystemConstructor(const std::string& system_urdf_filename,
//                         const std::string& plane_urdf_filename,
//                         bool is_floating, double joint_stiffness,
//                         double joint_damping)
//       : system_urdf_filename(system_urdf_filename),
//         plane_urdf_filename(plane_urdf_filename),
//         is_floating(is_floating),
//         joint_stiffness(joint_stiffness),
//         joint_damping(joint_damping) {}

//   // template <typename VisualizerAPI, typename Algebra>
//   // void operator()(VisualizerAPI* sim, VisualizerAPI* vis,
//   //                 World<Algebra>& world,
//   //                 MultiBody<Algebra>** system,
//   //                 bool clear_cache = false) const {
//   //   static UrdfCache<Algebra> cache;
//   //   if (clear_cache) {
//   //     cache.data.clear();
//   //   }

//   //   b3RobotSimulatorLoadUrdfFileArgs args;
//   //   args.m_flags |= URDF_MERGE_FIXED_LINKS;
//   //   if (!m_plane_urdf_filename.empty()) {
//   //     MultiBody<Algebra>* mb = world.create_multi_body();
//   //     const auto& urdf_data =
//   //         cache.retrieve(plane_urdf_filename, sim, vis, args);
//   //     UrdfToMultiBody<Algebra>::convert_to_multi_body(urdf_data,
//   //                                                               world,
//   *mb);
//   //   }

//   //   {
//   //     MultiBody<Algebra>* mb = world.create_multi_body();
//   //     const auto& urdf_data =
//   //         cache.retrieve(system_urdf_filename, sim, vis, args);
//   //     UrdfToMultiBody<Algebra>::convert_to_multi_body(urdf_data,
//   //                                                               world,
//   *mb);
//   //     mb->isFloating = is_floating;
//   //     if (!m_control_indices.empty()) {
//   //       mb->control_indices = control_indices;
//   //     }
//   //     mb->initialize();
//   //     if (actuator) {
//   //       mb->actuator = new Actuator<Algebra>(*m_actuator);
//   //     }
//   //     for (auto& link : mb->links) {
//   //       link.m_stiffness = Utils::scalar_from_double(joint_stiffness);
//   //       link.m_damping = Utils::scalar_from_double(joint_damping);
//   //     }
//   //     *system = mb;
//   //   }
//   // }

//   template <typename Algebra>
//   void operator()(World<Algebra>& world,
//                   MultiBody<Algebra>** system,
//                   bool clear_cache = false) const {
//     thread_local static UrdfCache<Algebra> cache;
//     if (clear_cache) {
//       cache.data.clear();
//     }

//     if (!m_plane_urdf_filename.empty()) {
//       MultiBody<Algebra>* mb = world.create_multi_body();
//       const auto& urdf_data = cache.retrieve(plane_urdf_filename);
//       UrdfToMultiBody<Algebra>::convert_to_multi_body(urdf_data,
//                                                                 world, *mb);
//     }

//     {
//       MultiBody<Algebra>* mb = world.create_multi_body();
//       const auto& urdf_data = cache.retrieve(system_urdf_filename);
//       UrdfToMultiBody<Algebra>::convert_to_multi_body(urdf_data,
//                                                                 world, *mb);
//       mb->isFloating = is_floating;
//       if (!m_control_indices.empty()) {
//         mb->control_indices = control_indices;
//       }
//       mb->initialize();
//       if (actuator) {
//         mb->actuator = new Actuator<Algebra>(*m_actuator);
//       }
//       for (auto& link : mb->links) {
//         link.m_stiffness = Utils::scalar_from_double(joint_stiffness);
//         link.m_damping = Utils::scalar_from_double(joint_damping);
//       }
//       *system = mb;
//     }
//   }
// };
}  // namespace tds
