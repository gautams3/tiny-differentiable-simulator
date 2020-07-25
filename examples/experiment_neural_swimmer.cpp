// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <fenv.h>
#include <iostream>
#include <string>
#include <thread>

#include "neural_augmentation.h"
#include "neural_scalar.h"
#include "tiny_ceres_estimator.h"
#include "tiny_dataset.h"
#include "tiny_double_utils.h"
#include "tiny_file_utils.h"
#include "tiny_multi_body.h"
#include "tiny_neural_network.h"
#include "tiny_urdf_parser.h"
#include "tiny_urdf_structures.h"
#include "tiny_urdf_to_multi_body.h"
#include "tiny_world.h"

constexpr int kJoints = 5;
constexpr double DT = 1. / 500.;
NeuralAugmentation augmentation;

// Load the structures from a URDF file. Split at the structures for better
// caching.
bool LoadURDFStructures(
    const std::string &urdf_filename,
    TinyUrdfStructures<double, DoubleUtils> *urdf_structures) {
  std::ifstream ifs((urdf_filename));
  std::string urdf_string;
  if (!ifs.is_open()) {
    std::cout << "Error, cannot open file_name: " << urdf_filename << std::endl;
    return false;
  }
  urdf_string = std::string(std::istreambuf_iterator<char>(ifs),
                            std::istreambuf_iterator<char>());
  int flags = 0;
  StdLogger logger;
  TinyUrdfParser<double, DoubleUtils> parser;
  parser.load_urdf_from_string(urdf_string, flags, logger, *urdf_structures);

  return true;
}

// Load a urdf from structure cache.
template <typename T, typename TUtils>
TinyMultiBody<T, TUtils> *
LoadURDF(TinyWorld<T, TUtils> *world,
         const TinyUrdfStructures<double, DoubleUtils> &urdf_structures) {
  TinyMultiBody<T, TUtils> *system = world->create_multi_body();
  system->m_isFloating = false;
  TinyUrdfToMultiBody<T, TUtils>::convert_to_multi_body(urdf_structures, *world,
                                                        *system);
  system->initialize();

  return system;
}

// Convenience class for reading the dataset.
struct DatasetRow {
  DatasetRow(double *data) : tau(data) {}
  double *tau;
  double *q;
  double *xyzpos;
  double *xyzvel;
};

// State used in trajectory rollouts.
template <typename T> struct State {
  State(const std::vector<T> &q, const std::vector<T> &qd) : q(q), qd(qd) {}
  std::vector<T> q;
  std::vector<T> qd;
};

// Rollout a swimmer given the system parameters, from a trajectory id specified
// on the first axis of the dataset.
template <typename T = double, typename TUtils = DoubleUtils>
void RolloutSwimmer(
    const TinyUrdfStructures<double, DoubleUtils> urdf_structures,
    const std::vector<T> &params, const TinyDataset<double, 3> &dataset,
    std::size_t traj_id, std::vector<State<T>> *output) {
  // Create the world and load the system.
  TinyWorld<T, TUtils> world;
  TinyMultiBody<T, TUtils> *system = LoadURDF(&world, urdf_structures);

  if constexpr (is_neural_scalar<T, TUtils>::value) {
    if (!params.empty()) {
      using InnerT = typename T::InnerScalarType;
      using InnerTUtils = typename T::InnerUtilsType;
      std::vector<InnerT> iparams(params.size());
      for (std::size_t i = 0; i < params.size(); ++i) {
        iparams[i] = params[i].evaluate();
      }
      augmentation.template instantiate<InnerT, InnerTUtils>(iparams);
    }
  }

  const std::size_t total_timesteps = dataset.Shape()[1];
  output->reserve(total_timesteps);
  for (std::size_t timestep = 0; timestep < total_timesteps; ++timestep) {
    // Set gravity.
    world.set_gravity(TinyVector3<double, DoubleUtils>(0, 0, 0));

    // Apply force from the dataset.
    for (std::size_t joint = 0; joint < kJoints; ++joint) {
      const std::array<std::size_t, 3> idx = {traj_id, timestep, joint};
      system->m_tau[3 + joint] = dataset[idx];
    }

    // Save state to output.
    output->push_back({system->m_q, system->m_qd});

    // Run dynamics.
    system->forward_dynamics(world.get_gravity());
    system->clear_forces();
    world.step(DT);
    system->integrate(TUtils::scalar_from_double(DT));
  }
}

int main(int argc, char *argv[]) {
  // Set NaN trap
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // Find urdf and dataset files.
  std::string dataset_filename, urdf_filename;
  TinyFileUtils::find_file("swimmer05.npy", dataset_filename);
  TinyFileUtils::find_file("swimmer/swimmer05/swimmer05.urdf", urdf_filename);

  // Load URDF cache.
  TinyUrdfStructures<double, DoubleUtils> urdf_structures;
  if (!LoadURDFStructures(urdf_filename, &urdf_structures)) {
    return 1;
  }

  // Load datasets.
  TinyNumpyReader<double, 3> reader;
  const bool status = reader.Open(dataset_filename);
  if (!status) {
    std::cerr << "Error reading dataset: " << reader.ErrorStatus() << "\n";
    return -1;
  }
  TinyDataset<double, 3> dataset = reader.Read();

  // Setup neural augmentation.
  augmentation.add_wiring();

  // Test rollout.
  std::vector<State<double>> rollout_trajectory_states;
  RolloutSwimmer(urdf_structures, {}, dataset, 0, &rollout_trajectory_states);

  return EXIT_SUCCESS;
}
