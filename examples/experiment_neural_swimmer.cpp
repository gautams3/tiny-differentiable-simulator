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

#include <cassert>
#include <ceres/loss_function.h>
#define NEURAL_SIM 1

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

constexpr bool kUsePBH = true;
constexpr int kLinks = 5;
constexpr int kJoints = kLinks - 1;
constexpr double kDT = 1. / 500.;
constexpr int kTimeSteps = 201;
constexpr int kParamDim = 83;
constexpr int kStateDim = (3 + 3) * kLinks; // 3 pos + 3 vel per link

NeuralAugmentation gAugmentation;

// Cache for URDF Structures, especially in case we need to load multiple types.
template <typename T, typename TUtils> struct UrdfCache {
  using UrdfStructures = TinyUrdfStructures<T, TUtils>;
  static thread_local inline std::map<std::string, UrdfStructures> cache;

  // Load a structure from the cache, by filename.
  static const UrdfStructures &Get(const std::string &urdf_filename) {
    if (cache.find(urdf_filename) == cache.end()) {
      std::string real_filename;
      TinyFileUtils::find_file(urdf_filename, real_filename);
      printf("Loading URDF \"%s\".\n", real_filename.c_str());
      TinyUrdfParser<T, TUtils> parser;
      cache[urdf_filename] = parser.load_urdf(real_filename);
    }
    return cache[urdf_filename];
  }
};

// Cache for Datasets.
struct DatasetCache {
  using Dataset = TinyDataset<double, 3>;
  using DatasetAsVectors =
      std::pair<std::vector<std::vector<double>>,
                std::vector<std::vector<std::vector<double>>>>;

  static thread_local inline std::map<std::string, Dataset> cache;
  static thread_local inline std::map<std::string, DatasetAsVectors>
      cache_as_vectors;

  // Load a structure from the cache, by filename.
  static const Dataset &Get(const std::string &dataset_filename) {
    if (cache.find(dataset_filename) == cache.end()) {
      std::string real_filename;
      TinyFileUtils::find_file(dataset_filename, real_filename);
      printf("Loading dataset \"%s\".\n", real_filename.c_str());
      TinyNumpyReader<double, 3> reader;
      const bool status = reader.Open(real_filename);
      if (!status) {
        std::cerr << "Error reading dataset: " << reader.ErrorStatus() << "\n";
      }
      cache[dataset_filename] = reader.Read();
    }
    return cache[dataset_filename];
  }

  static const DatasetAsVectors &
  GetAsVectors(const std::string &dataset_filename) {
    if (cache_as_vectors.find(dataset_filename) == cache_as_vectors.end()) {
      const Dataset &dataset = Get(dataset_filename);
      const auto [ntraj, ntimesteps, nstate] = dataset.Shape();
      std::vector<std::vector<double>> times;
      std::vector<std::vector<std::vector<double>>> states;

      times.resize(ntraj);
      states.resize(ntraj);
      for (std::size_t traj = 0; traj < ntraj; ++traj) {
        // Leave the times as empty vectors, since the timesteps match.
        states[traj].resize(ntimesteps);
        for (std::size_t timestep = 0; timestep < ntimesteps; ++timestep) {
          states[traj][timestep].resize(nstate);
          for (std::size_t i = 0; i < nstate; ++i) {
            std::array<std::size_t, 3> idx = {traj, timestep, i};
            states[traj][timestep][i] = dataset[idx];
          }
        }
      }
      cache_as_vectors[dataset_filename] = {times, states};
    }
    return cache_as_vectors[dataset_filename];
  }
};

// Load a urdf from structure cache.
template <typename T, typename TUtils>
TinyMultiBody<T, TUtils> *
LoadURDF(TinyWorld<T, TUtils> *world,
         const TinyUrdfStructures<T, TUtils> &urdf_structures) {
  TinyMultiBody<T, TUtils> *system = world->create_multi_body();
  system->m_isFloating = false;
  TinyUrdfToMultiBody<T, TUtils>::convert_to_multi_body(urdf_structures, *world,
                                                        *system);
  system->initialize();

  return system;
}

// Rollout a swimmer given the system parameters, from a trajectory id
// specified on the first axis of the dataset.
template <typename T = double, typename TUtils = DoubleUtils>
void RolloutSwimmer(const TinyUrdfStructures<T, TUtils> urdf_structures,
                    const std::vector<T> &params,
                    const TinyDataset<double, 3> &dataset, std::size_t traj_id,
                    std::size_t requested_timesteps, double *dt,
                    std::vector<std::vector<T>> *output) {
  // Create the world and load the system.
  TinyWorld<T, TUtils> world;
  TinyMultiBody<T, TUtils> *system = LoadURDF(&world, urdf_structures);
  *dt = kDT;

  if constexpr (is_neural_scalar<T, TUtils>::value) {
    if (!params.empty()) {
      using InnerT = typename T::InnerScalarType;
      using InnerTUtils = typename T::InnerUtilsType;
      std::vector<InnerT> iparams(params.size());
      for (std::size_t i = 0; i < params.size(); ++i) {
        iparams[i] = params[i].evaluate();
      }
      gAugmentation.template instantiate<InnerT, InnerTUtils>(iparams);
    }
  }

  const std::size_t dataset_timesteps = dataset.Shape()[1];
  if (requested_timesteps > dataset_timesteps) {
    std::cerr << "ERROR! Requested more timesteps than in dataset.\n";
    return;
  }

  output->reserve(requested_timesteps);
  for (std::size_t timestep = 0; timestep < requested_timesteps; ++timestep) {
    // Set gravity.
    world.set_gravity(TinyVector3<T, TUtils>::zero());

    // Apply force from the dataset.
    for (std::size_t joint = 0; joint < kJoints; ++joint) {
      const std::array<std::size_t, 3> idx = {traj_id, timestep, joint};
      system->m_tau[3 + joint] = TUtils::scalar_from_double(dataset[idx]);
    }

    // Save state to output.
    std::vector<T> state;
    state.reserve(kStateDim);
    for (const TinyLink<T, TUtils> &link : system->m_links) {
      state.push_back(link.m_X_world.m_translation[0]); // pos_x
      state.push_back(link.m_X_world.m_translation[1]); // pos_y
      state.push_back(
          TUtils::atan2(link.m_X_world.m_rotation(0, 1),
                        link.m_X_world.m_rotation(0, 0))); // pos_yaw
      state.push_back(link.m_v[3]);                        // vel_x
      state.push_back(link.m_v[4]);                        // vel_y
      state.push_back(link.m_v[4]);                        // vel_yaw
    }
    output->push_back(state);

    // Run dynamics.
    system->forward_dynamics(world.get_gravity());
    system->clear_forces();
    world.step(TUtils::scalar_from_double(*dt));
    system->integrate(TUtils::scalar_from_double(*dt));
  }
}

class SwimmerEstimator
    : public TinyCeresEstimator<kParamDim, kStateDim, RES_MODE_1D> {
public:
  typedef TinyCeresEstimator<kParamDim, kStateDim, RES_MODE_1D> CeresEstimator;
  using CeresEstimator::kStateDim, CeresEstimator::kParameterDim;
  using CeresEstimator::parameters;
  using typename CeresEstimator::ADScalar;

  int timesteps_;
  std::string urdf_filename_;
  std::string dataset_filename_;

  SwimmerEstimator(const std::string &urdf_filename,
                   const std::string &dataset_filename)
      : CeresEstimator(kDT), timesteps_(kTimeSteps),
        urdf_filename_(urdf_filename), dataset_filename_(dataset_filename) {
    gAugmentation.assign_estimation_parameters(parameters);
  }

  static std::function<std::unique_ptr<SwimmerEstimator>()>
  Factory(const std::string &urdf_filename,
          const std::string &dataset_filename) {
    return [&urdf_filename, &dataset_filename]() {
      auto estimator =
          std::make_unique<SwimmerEstimator>(urdf_filename, dataset_filename);
      auto [target_times, target_states] =
          DatasetCache::GetAsVectors(dataset_filename);
      estimator->target_times = target_times;
      estimator->target_trajectories = target_states;
      estimator->options.minimizer_progress_to_stdout = !kUsePBH;
      estimator->options.max_num_consecutive_invalid_steps = 100;
      estimator->divide_cost_by_time_factor = 10.;
      estimator->divide_cost_by_time_exponent = 1.2;
      return estimator;
    };
  }

  void rollout(const std::vector<ADScalar> &params,
               std::vector<std::vector<ADScalar>> &output_states, double &dt,
               std::size_t ref_id) const override {
    typedef CeresUtils<kParameterDim> ADUtils;
    typedef NeuralScalar<ADScalar, ADUtils> NScalar;
    typedef NeuralScalarUtils<ADScalar, ADUtils> NUtils;
    auto n_params = NUtils::to_neural(params);
    std::vector<std::vector<NScalar>> n_output_states;
    RolloutSwimmer<NScalar, NUtils>(
        UrdfCache<NScalar, NUtils>::Get(urdf_filename_), n_params,
        DatasetCache::Get(dataset_filename_), ref_id, timesteps_, &dt,
        &n_output_states);
    for (const auto &state : n_output_states) {
      output_states.push_back(NUtils::from_neural(state));
    }
  }
  void rollout(const std::vector<double> &params,
               std::vector<std::vector<double>> &output_states, double &dt,
               std::size_t ref_id) const override {
    typedef NeuralScalar<double, DoubleUtils> NScalar;
    typedef NeuralScalarUtils<double, DoubleUtils> NUtils;
    auto n_params = NUtils::to_neural(params);
    std::vector<std::vector<NScalar>> n_output_states;
    RolloutSwimmer<NScalar, NUtils>(
        UrdfCache<NScalar, NUtils>::Get(urdf_filename_), n_params,
        DatasetCache::Get(dataset_filename_), ref_id, timesteps_, &dt,
        &n_output_states);
    for (const auto &state : n_output_states) {
      output_states.push_back(NUtils::from_neural(state));
    }
  }
};

int main(int argc, char *argv[]) {
  // Filenames.
  const std::string urdf_filename = "swimmer/swimmer05/swimmer05.urdf";
  const std::string dataset_filename = "swimmer/swimmer05.npy";

  // Setup neural augmentation.
  std::vector<std::string> inputs;
  for (const std::string &info : {"pos", "vel"}) {
    for (const std::string &dim : {"x", "y", "yaw"}) {
      inputs.push_back("link/" + info + "/" + dim);
    }
  }
  std::vector<std::string> outputs;
  for (const std::string &info : {"external_force"}) {
    for (const std::string &dim : {"x", "y", "yaw"}) {
      outputs.push_back("link/" + info + "/" + dim);
    }
  }
  gAugmentation.add_wiring(outputs, inputs);
  std::cout << "#params = " << gAugmentation.num_total_parameters() << "\n";

  auto estimator_factory =
      SwimmerEstimator::Factory(urdf_filename, dataset_filename);
  std::vector<double> best_params;

  // Run estimator.
  if (kUsePBH) {
    // If we use parallel basin hopping.
    std::array<double, kParamDim> initial_guess;
    for (int i = 0; i < kParamDim; ++i) {
      initial_guess[i] = 0.0;
    }
    BasinHoppingEstimator<kParamDim, SwimmerEstimator> bhe(estimator_factory,
                                                           initial_guess);
    bhe.time_limit = 60 * 60 * 4;
    bhe.run();

    printf("Best cost: %f\n", bhe.best_cost());
    best_params.insert(best_params.end(), bhe.params.begin(), bhe.params.end());
    for (const double param : bhe.params) {
      best_params.push_back(param);
    }
  } else {
    // If we use don't parallel basin hopping.
    std::unique_ptr<SwimmerEstimator> estimator = estimator_factory();
    estimator->setup(new ceres::HuberLoss(1.));
    auto summary = estimator->solve();
    std::cout << summary.FullReport() << "\n";
    std::cout << "Final cost:" << summary.final_cost << "\n";
    for (const EstimationParameter &param : estimator->parameters) {
      best_params.push_back(param.value);
    }
  }

  printf("Optimized parameters:");
  for (const double param : best_params) {
    printf(" %.8f", param);
  }

  return EXIT_SUCCESS;
}
