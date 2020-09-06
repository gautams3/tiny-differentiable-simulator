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

#include "../multi_body.hpp"
#include "urdf_parser.hpp"
#include "urdf_to_multi_body.hpp"
#include "pybullet_urdf_import.hpp"
#include "world.hpp"

namespace tds {
template <typename Algebra>
struct UrdfCache {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  typedef tds::UrdfStructures<Algebra> UrdfStructures;

  typedef tds::PyBulletUrdfImport<Algebra> UrdfImport;
  typedef b3RobotSimulatorLoadUrdfFileArgs UrdfFileArgs;

  std::map<std::string, UrdfStructures> data;

  template <typename VisualizerAPI>
  const UrdfStructures& retrieve(const std::string& urdf_filename,
                                 VisualizerAPI* sim, VisualizerAPI* vis,
                                 UrdfFileArgs args = UrdfFileArgs(),
                                 bool ignore_cache = false) {
    if (ignore_cache || data.find(urdf_filename) == data.end()) {
      printf("Loading URDF \"%s\".\n", urdf_filename.c_str());
      int robotId = sim->loadURDF(urdf_filename, args);
      if (robotId < 0) {
        std::cerr << "Error: Could not load URDF file \"" << urdf_filename
                  << "\".\n";
        exit(1);
      }
      data[urdf_filename] = UrdfStructures();
      UrdfImport::extract_urdf_structs(data[urdf_filename], robotId, *sim,
                                       *vis);
      sim->removeBody(robotId);
    }
    return data[urdf_filename];
  }

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
                                World<Algebra>& world,
                                bool ignore_cache = false,
                                bool is_floating = false) {
    MultiBody<Algebra>* mb = world.create_multi_body();
    const auto& urdf_data = retrieve(urdf_filename, ignore_cache);
    UrdfToMultiBody<Algebra>::convert_to_multi_body(urdf_data, world, *mb);
    mb->set_floating_base(is_floating);
    mb->initialize();
    return mb;
  }

  template <typename VisualizerAPI>
  MultiBody<Algebra>* construct(const std::string& urdf_filename,
                                          World<Algebra>& world,
                                          VisualizerAPI* sim,
                                          VisualizerAPI* vis,
                                          bool ignore_cache = false,
                                          bool is_floating = false) {
    b3RobotSimulatorLoadUrdfFileArgs args;
    args.m_flags |= URDF_MERGE_FIXED_LINKS;
    MultiBody<Algebra>* mb = world.create_multi_body();
    const auto& urdf_data =
        retrieve(urdf_filename, sim, vis, args, ignore_cache);
    UrdfToMultiBody<Algebra>::convert_to_multi_body(urdf_data, world,
                                                              *mb);
    mb->isFloating = is_floating;
    mb->initialize();
    return mb;
  }
};

}  // namespace tds
