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

#include <ceres/loss_function.h>
#include <ceres/types.h>

#include <cassert>

#include "math/tiny/tiny_algebra.hpp"
#define NEURAL_SIM 1

#include <fenv.h>

#include <iostream>
#include <string>
#include <thread>

#include "dynamics/forward_dynamics.hpp"
#include "dynamics/integrator.hpp"
#include "dynamics/kinematics.hpp"
#include "math/neural_network.hpp"
#include "math/tiny/neural_scalar.hpp"
#include "math/tiny/tiny_double_utils.h"
#include "multi_body.hpp"
#include "neural_augmentation.h"
#include "tiny_ceres_estimator.h"
#include "urdf/urdf_parser.hpp"
#include "urdf/urdf_structures.hpp"
#include "urdf/urdf_to_multi_body.hpp"
#include "utils/file_utils.hpp"
#include "utils/tiny_dataset.h"
#include "world.hpp"

constexpr bool kUsePBH = false;
constexpr bool kStopAfterSetup = false;
constexpr bool kStateHasVel = true;

constexpr int kLinks = 5;
constexpr int kJoints = kLinks - 1;
constexpr double kDT = 0.002;
constexpr int kTimesteps = 201;
constexpr int kParamDim = 83;
constexpr int kStateDim = (kStateHasVel ? 6 : 3) * kLinks;

// Some info about how the dataset is laid out.
constexpr std::size_t kControlTimesteps = 10;
constexpr std::size_t kTauOffset = 0;
constexpr std::size_t kQOffset = kTauOffset + kJoints;
constexpr std::size_t kPosOffset = kQOffset + kJoints;
constexpr std::size_t kVelOffset = kPosOffset + 3 * kLinks;

// Some info about our URDFs.
constexpr std::size_t kUrdfJoint0Offset = 3;  // x, y, yaw come first
constexpr std::size_t kUrdfLink0Offset = 2;   // xslide, yslide

// Global neural augmentation tracker.
tds::NeuralAugmentation gAugmentation;

// Cache for URDF Structures, especially in case we need to load multiple types.
template <typename Algebra>
struct UrdfCache {
  using UrdfStructures = tds::UrdfStructures<Algebra>;
  static thread_local inline std::map<std::string, UrdfStructures> cache;

  // Load a structure from the cache, by filename.
  static const UrdfStructures &Get(const std::string &urdf_filename) {
    if (cache.find(urdf_filename) == cache.end()) {
      std::string real_filename;
      tds::FileUtils::find_file(urdf_filename, real_filename);
      printf("Loading URDF \"%s\".\n", real_filename.c_str());
      tds::UrdfParser<Algebra> parser;
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
      tds::FileUtils::find_file(dataset_filename, real_filename);
      printf("Loading dataset \"%s\".\n", real_filename.c_str());
      TinyNumpyReader<double, 3> reader;
      const bool status = reader.Open(real_filename);
      if (!status) {
        std::cerr << "Error reading dataset: " << reader.ErrorStatus() << "\n";
        std::exit(1);
      }
      cache[dataset_filename] = reader.Read();
    }
    return cache[dataset_filename];
  }

  static const DatasetAsVectors &GetAsVectors(
      const std::string &dataset_filename) {
    if (cache_as_vectors.find(dataset_filename) == cache_as_vectors.end()) {
      const Dataset &dataset = Get(dataset_filename);
      const auto [ntraj, ntimesteps, nstatedataset] = dataset.Shape();
      std::vector<std::vector<double>> times;
      std::vector<std::vector<std::vector<double>>> states;

      times.resize(ntraj);
      states.resize(ntraj);
      for (std::size_t traj = 0; traj < ntraj; ++traj) {
        // Leave the times as empty vectors, since the timesteps match.
        states[traj].resize(kTimesteps);
        for (std::size_t timestep = 0; timestep < kTimesteps; ++timestep) {
          states[traj][timestep].resize(kStateDim);
          for (std::size_t i = 0; i < kStateDim; ++i) {
            std::array<std::size_t, 3> idx = {traj, timestep, kPosOffset + i};
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
template <typename Algebra>
tds::MultiBody<Algebra> *LoadURDF(
    tds::World<Algebra> *world,
    const tds::UrdfStructures<Algebra> &urdf_structures) {
  tds::MultiBody<Algebra> *system = world->create_multi_body();
  system->set_floating_base(false);
  tds::UrdfToMultiBody<Algebra>::convert_to_multi_body(urdf_structures, *world,
                                                       *system);
  system->initialize();

  return system;
}

// Rollout a swimmer given the system parameters, from a trajectory id
// specified on the first axis of the dataset.
template <typename Algebra = tds::DoubleAlgebra>
void RolloutSwimmer(const tds::UrdfStructures<Algebra> urdf_structures,
                    const std::vector<typename Algebra::Scalar> &params,
                    const TinyDataset<double, 3> &dataset, std::size_t traj_id,
                    std::size_t requested_timesteps, double *dt,
                    std::vector<std::vector<typename Algebra::Scalar>> *output,
                    std::vector<std::vector<double>> *full_qs_debug = nullptr) {
  if (full_qs_debug != nullptr) {
    full_qs_debug->clear();
  }

  // Create the world and load the system.
  tds::World<Algebra> world;
  tds::MultiBody<Algebra> *system = LoadURDF(&world, urdf_structures);
  *dt = kDT * kControlTimesteps;

  // Initialize the neural network from the parameters.
  if constexpr (tds::is_neural_algebra<Algebra>::value) {
    using InnerAlgebra = typename Algebra::Scalar::InnerAlgebra;
    if (!params.empty()) {
      std::vector<typename InnerAlgebra::Scalar> iparams(params.size());
      for (std::size_t i = 0; i < params.size(); ++i) {
        iparams[i] = params[i].evaluate();
      }
      gAugmentation.template instantiate<InnerAlgebra>(iparams);
    }
  }

  // Some sanity checking for the dataset.
  const std::size_t dataset_timesteps = dataset.Shape()[1];
  if (requested_timesteps > dataset_timesteps) {
    std::cerr << "ERROR! Requested more timesteps than in dataset.\n";
    std::exit(1);
  }

  // Set the initial state from the data file.
  // Global yaw.
  const std::array<std::size_t, 3> idx = {traj_id, 0, kPosOffset + 2};
  system->q()[kUrdfJoint0Offset - 1] = Algebra::from_double(dataset[idx]);
  // Full Q.
  for (int joint = 0; joint < kJoints; ++joint) {
    const std::array<std::size_t, 3> idx = {traj_id, 0, kQOffset + joint};
    system->q()[kUrdfJoint0Offset + joint] = Algebra::from_double(dataset[idx]);
  }
  // Update forward kinematics.
  forward_kinematics(*system);

  // Run the actual rollout.
  output->reserve(requested_timesteps);
  for (std::size_t timestep = 0; timestep < requested_timesteps; ++timestep) {
    // Save state to output.
    std::vector<typename Algebra::Scalar> state;
    state.reserve(kStateDim);
    for (std::size_t i = 0; i < kLinks; ++i) {
      const tds::Link<Algebra> &link = system->links()[kUrdfLink0Offset + i];
      state.push_back(link.X_world.translation[0]);  // pos_x
      state.push_back(link.X_world.translation[1]);  // pos_y
      state.push_back(Algebra::atan2(-link.X_world.rotation(0, 1),
                                     link.X_world.rotation(0, 0)));  // pos_yaw
    }
    if (kStateHasVel) {
      for (std::size_t i = 0; i < kLinks; ++i) {
        const tds::Link<Algebra> &link = system->links()[kUrdfLink0Offset + i];
        state.push_back(link.v[3]);  // vel_x
        state.push_back(link.v[4]);  // vel_y
        state.push_back(link.v[2]);  // vel_yaw
      }
    }
    output->push_back(state);

    // Run dynamics for the number of control timesteps.
    for (std::size_t control_timestep = 0; control_timestep < kControlTimesteps;
         ++control_timestep) {
      // Apply force from the dataset.
      for (std::size_t joint = 0; joint < kJoints; ++joint) {
        const std::array<std::size_t, 3> idx = {traj_id, timestep,
                                                kTauOffset + joint};
        system->tau()[kUrdfJoint0Offset + joint] =
            Algebra::from_double(dataset[idx]);
      }

      if (Algebra::to_double(system->tau()[0]) != 0 ||
          Algebra::to_double(system->tau()[1]) != 0 ||
          Algebra::to_double(system->tau()[2]) != 0) {
        std::cout << "\n\nActuating unactuated joint!\n\n";
        std::exit(1);
      }

      if (full_qs_debug != nullptr) {
        std::vector<double> full_q;
        for (int i = 0; i < system->q().m_size; ++i) {
          full_q.push_back(Algebra::to_double(system->q()[i]));
        }
        full_qs_debug->push_back(full_q);
      }

      forward_dynamics(*system, Algebra::zero3());
      system->clear_forces();
      integrate_euler(*system, Algebra::from_double(*dt));
    }
  }
}

class SwimmerEstimator
    : public TinyCeresEstimator<kParamDim, kStateDim, RES_MODE_1D> {
 public:
  using CeresEstimator = TinyCeresEstimator<kParamDim, kStateDim, RES_MODE_1D>;
  using CeresEstimator::kStateDim, CeresEstimator::kParameterDim;
  using CeresEstimator::parameters;
  using typename CeresEstimator::ADScalar;

  int timesteps_;
  std::string urdf_filename_;
  std::string dataset_filename_;

  SwimmerEstimator(const std::string &urdf_filename,
                   const std::string &dataset_filename)
      : CeresEstimator(kDT),
        timesteps_(kTimesteps),
        urdf_filename_(urdf_filename),
        dataset_filename_(dataset_filename) {
    gAugmentation.assign_estimation_parameters(parameters);
    for (std::size_t i = 0; i < kParameterDim; ++i) {
      parameters[i].value *= 0.01;  // Scale down to avoid NaNs immediately.
    }
  }

  static std::function<std::unique_ptr<SwimmerEstimator>()> Factory(
      const std::string &urdf_filename, const std::string &dataset_filename) {
    return [&urdf_filename, &dataset_filename]() {
      auto estimator =
          std::make_unique<SwimmerEstimator>(urdf_filename, dataset_filename);
      auto [target_times, target_states] =
          DatasetCache::GetAsVectors(dataset_filename);
      estimator->target_times = target_times;
      estimator->target_trajectories = target_states;
      estimator->options.minimizer_progress_to_stdout = !kUsePBH;
      estimator->options.minimizer_type = ceres::MinimizerType::LINE_SEARCH;
      estimator->set_bounds = false;
      estimator->options.max_num_consecutive_invalid_steps = 100;
      estimator->divide_cost_by_time_factor = 0.;
      estimator->divide_cost_by_time_exponent = 1.2;
      return estimator;
    };
  }

  void rollout(const std::vector<ADScalar> &params,
               std::vector<std::vector<ADScalar>> &output_states, double &dt,
               std::size_t ref_id) const override {
    using CeresAlgebra = TinyAlgebra<ADScalar, CeresUtils<kParameterDim>>;
    using NCAlgebra = tds::NeuralAlgebra<CeresAlgebra>;
    auto n_params = tds::to_neural<NCAlgebra>(params);
    std::vector<std::vector<NCAlgebra::Scalar>> n_output_states;
    RolloutSwimmer<NCAlgebra>(UrdfCache<NCAlgebra>::Get(urdf_filename_),
                              n_params, DatasetCache::Get(dataset_filename_),
                              ref_id, timesteps_, &dt, &n_output_states);
    for (const auto &state : n_output_states) {
      output_states.push_back(tds::from_neural<NCAlgebra>(state));
    }
  }
  void rollout(const std::vector<double> &params,
               std::vector<std::vector<double>> &output_states, double &dt,
               std::size_t ref_id) const override {
    using NAlgebra = tds::NeuralAlgebra<tds::DoubleAlgebra>;
    auto n_params = tds::to_neural<NAlgebra>(params);
    std::vector<std::vector<NAlgebra::Scalar>> n_output_states;
    RolloutSwimmer<NAlgebra>(UrdfCache<NAlgebra>::Get(urdf_filename_), n_params,
                             DatasetCache::Get(dataset_filename_), ref_id,
                             timesteps_, &dt, &n_output_states);
    for (const auto &state : n_output_states) {
      output_states.push_back(tds::from_neural<NAlgebra>(state));
    }
  }
};

void WriteRollout(const std::string &output_prefix,
                  const std::string &urdf_filename,
                  const std::string &dataset_filename,
                  const std::vector<double> &params) {
  double rollout_dt;
  std::vector<std::vector<double>> states;
  std::vector<std::vector<double>> full_qs_debug;
  RolloutSwimmer<tds::DoubleAlgebra>(
      UrdfCache<tds::DoubleAlgebra>::Get(urdf_filename), params,
      DatasetCache::Get(dataset_filename), 0, kTimesteps, &rollout_dt, &states,
      &full_qs_debug);

  {
    const std::string &rollout_filename = output_prefix + "_rollout.csv";
    std::cout << "\n\n";
    std::cout << "Writing rollout to " << rollout_filename << ", "
              << states.size() << " timesteps\n";
    std::ofstream f(rollout_filename);
    for (const std::vector<double> &state : states) {
      bool first = true;
      for (double statei : state) {
        f << (first ? "" : ",") << statei;
        first = false;
      }
      f << "\n";
      std::cout << ".";
      std::cout.flush();
    }
    f.flush();
    std::cout << "\n";
  }

  {
    const std::string &qs_filename = output_prefix + "_qs.csv";
    std::cout << "\n\n";
    std::cout << "Writing full qs to " << qs_filename << ", "
              << full_qs_debug.size() << " timesteps\n";
    std::ofstream f(qs_filename);
    for (const std::vector<double> &full_q : full_qs_debug) {
      bool first = true;
      for (double qi : full_q) {
        f << (first ? "" : ",") << qi;
        first = false;
      }
      f << "\n";
      std::cout << ".";
      std::cout.flush();
    }
    f.flush();
    std::cout << "\n";
  }

  {
    const std::string &params_filename = output_prefix + "_params.txt";
    std::cout << "Writing params to " << params_filename << "\n";
    std::ofstream f(params_filename);
    for (double param : params) {
      f << param << "\n";
      std::cout << ".";
      std::cout.flush();
    }
    f.flush();
    std::cout << "\n\n";
  }
}

int main(int argc, char *argv[]) {
  // Filenames.
  const std::string urdf_filename = "swimmer/swimmer05/swimmer05.urdf";
  const std::string dataset_filename = "swimmer/swimmer05.npy";

  // Write some sanity check rollout.
  const std::vector<double> all_zero_params(kParamDim, 0.0);
  WriteRollout("all_zero", urdf_filename, dataset_filename, all_zero_params);

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
  gAugmentation.weight_limit = 0.01;
  gAugmentation.bias_limit = 0.001;
  if (gAugmentation.num_total_parameters() != kParamDim) {
    std::cerr << "ERROR: Param dim mismatch!\n";
    std::exit(1);
  }

  // Estimator factory for PBH.
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
    bhe.time_limit = 3 * 60 * 60;
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

    // Print an initial loss to make sure gradient isn't all zero.
    double cost;
    double params[kParamDim];
    double gradient[kParamDim];
    for (std::size_t i = 0; i < kParamDim; ++i) {
      params[i] = 0;
      gradient[i] = 0;
    }
    estimator->compute_loss(params, &cost, gradient);
    {
      std::ofstream f("all_zero_loss_info.txt");
      f << "\n\ninitial loss: " << cost << "\ninitial_gradient: ";
      for (std::size_t i = 0; i < kParamDim; ++i) {
        f << gradient[i] << "\t";
      }
      f << "\n\n";
    }

    if (kStopAfterSetup) {
      std::cout << "\n\nStopping early as requested in kStopAfterSetup.\n\n";
      std::exit(0);
    }

    // Run the optimization.
    auto summary = estimator->solve();
    std::cout << summary.FullReport() << "\n";
    std::cout << "Final cost:" << summary.final_cost << "\n";
    for (const EstimationParameter &param : estimator->parameters) {
      best_params.push_back(param.value);
    }

    // Write the final rollout to a file.
    WriteRollout("optimized", urdf_filename, dataset_filename, best_params);
  }

  printf("Optimized parameters:");
  for (const double param : best_params) {
    printf(" %.8f", param);
  }

  return EXIT_SUCCESS;
}
