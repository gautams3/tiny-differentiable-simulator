#ifndef NEURAL_SCALAR_H
#define NEURAL_SCALAR_H

#include <iostream>
#include <map>
#include <thread>

#include "../tiny_neural_network.h"

/**
 * Implements a "neural network" scalar type that accepts input connections from
 * other NeuralScalars. The scalar can either be evaluated as residual module,
 * where the output of the neural network is combined with the value of the
 * scalar, or computed solely by the neural network ignoring the scalar's stored
 * value.
 */
template <typename Scalar, typename Utils>
class NeuralScalar {
 public:
  typedef ::TinyNeuralNetwork<Scalar, Utils> NeuralNetworkType;
  typedef Scalar InnerScalarType;
  typedef Utils InnerUtilsType;

 private:
  /**
   * Value assigned from outside.
   */
  Scalar value_{Utils::zero()};

  /**
   * Cached value from last evaluation.
   */
  mutable Scalar cache_;

  /**
   * Whether evaluation is necessary, or the cached value can be returned.
   */
  mutable bool is_dirty_{true};

  // mutable std::vector<const NeuralScalar*> inputs_;

  // mutable NeuralNetworkType net_;

  /**
   * Neural scalars with the same name reuse the same neural network inputs,
   * parameters. No sharing takes place if the name is empty.
   */
  mutable std::string name_;

  /**
   * Index of blueprint whose output is used by this scalar.
   */
  mutable int blueprint_id_{-1};

  /**
   * Index in output vector of neural network that is connected to this scalar.
   */
  mutable std::size_t output_id_{0};

  /**
   * This blueprint allows the user to specify neural network inputs and weights
   * to be used once a NeuralScalar with the given name is created.
   */
  struct NeuralBlueprint {
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    NeuralNetworkType net;

    std::vector<Scalar> output_cache;
    mutable bool is_dirty{true};
  };

  struct GlobalData {
    std::map<std::string, const NeuralScalar*> named_scalars;

    // map scalar name to the indices of blueprints where it appears as input
    std::map<std::string, std::vector<int>> scalar_input_to_blueprint;
    // map scalar name to the index of blueprint where it appears as output
    std::map<std::string, int> scalar_to_blueprint;
    std::vector<NeuralBlueprint> blueprints;

    void clear() {
      named_scalars.clear();
      scalar_input_to_blueprint.clear();
      scalar_to_blueprint.clear();
      blueprints.clear();
    }
  };

  static inline std::map<std::thread::id, GlobalData> data_{};

  static GlobalData& get_data_() {
    auto id = std::this_thread::get_id();
    if (data_.find(id) == data_.end()) {
      data_[id] = GlobalData();
    }
    return data_[id];
  }

  static const Scalar& evaluate_network_(int network_id,
                                         std::size_t output_id) {
    assert(network_id >= 0);

    auto& data = get_data_();
    auto& blueprint = data.blueprints[network_id];
    if (blueprint.is_dirty) {
      std::vector<Scalar> input(blueprint.input_names.size());
      for (std::size_t i = 0; i < blueprint.input_names.size(); ++i) {
        input[i] = data.named_scalars[blueprint.input_names[i]]->evaluate();
      }
      blueprint.net.compute(input, blueprint.output_cache);
      blueprint.is_dirty = false;
    }
    return blueprint.output_cache[output_id];
  }

 public:
  /**
   * Whether the internal value is added to, or replaced by, the neural
   * network's output.
   */
  bool is_residual{true};

  NeuralScalar() = default;

  inline NeuralScalar(const Scalar& value) : value_(value) { is_dirty_ = true; }

  // NeuralScalar(const std::vector<NeuralScalar*>& inputs,
  //              bool use_input_bias = true)
  //     : inputs_(inputs),
  //       net_(static_cast<int>(inputs.size()), use_input_bias) {}
  // NeuralScalar(const std::vector<NeuralScalar*>& inputs,
  //              const TinyNeuralNetworkSpecification& spec)
  //     : inputs_(inputs), net_(spec) {}

  // implement custom assignment operator to prevent internal data get wiped
  // by unwanted overwrite with a copy-constructed object
  NeuralScalar& operator=(const NeuralScalar& rhs) {
    value_ = rhs.evaluate();

    is_dirty_ = true;
    return *this;
  }
  NeuralScalar& operator=(const Scalar& rhs) {
    value_ = rhs;
    is_dirty_ = true;
    return *this;
  }

  // const NeuralNetworkType& net() const { return net_; }
  // NeuralNetworkType& net() { return net_; }

  /**
   * Add input connection to this neural network.
   */
  // void connect(NeuralScalar* scalar,
  //              TinyNeuralNetworkActivation activation = NN_ACT_IDENTITY) {
  //   inputs_.push_back(scalar);
  //   net_.set_input_dim(net_.input_dim() + 1);
  //   // add output layer (if none has been created yet)
  //   if (net_.num_layers() == 1) {
  //     net_.add_linear_layer(activation, 1);
  //   }
  //   initialize();
  //   set_dirty();
  // }

  // void initialize(
  //     TinyNeuralNetworkInitialization init_method = NN_INIT_XAVIER) {
  //   net_.initialize(init_method);
  // }

  bool is_dirty() const { return is_dirty_; }
  void set_dirty() {
    if (blueprint_id_ >= 0) {
      is_dirty_ = true;
    }
  }

  /**
   * Retrieves neural network scalar by name, returns nullptr if no scalar with
   * such name exists.
   */
  static const NeuralScalar* retrieve(const std::string& name) {
    const auto& named_scalars = get_data_().named_scalars;
    if (named_scalars.find(name) == named_scalars.end()) {
      return nullptr;
    }
    return named_scalars.at(name);
  }

  /**
   * Assigns a name to this scalar and looks up whether any blueprint
   * for this scalar has been defined to set up input connections and the neural
   * network.
   */
  void assign(const std::string& name) const {
    name_ = name;
    GlobalData& data = get_data_();
    if (data.scalar_to_blueprint.find(name) != data.scalar_to_blueprint.end()) {
      // std::cerr << "NeuralScalar named \"" << name
      //           << "\" was assigned a name but has no registered
      //           blueprint.\n";
      // assert(0);
      // return;
      blueprint_id_ = data.scalar_to_blueprint[name];
      auto& blueprint = data.blueprints[blueprint_id_];
      // determine the index in the output vector of the network
      bool found = false;
      for (std::size_t i = 0; i < blueprint.output_names.size(); ++i) {
        if (blueprint.output_names[i] == name) {
          output_id_ = i;
          found = true;
          break;
        }
      }
      if (!found) {
        std::cerr << "Error: could not determine output vector index for "
                     "neural scalar \""
                  << name << "\" that was assigned a blueprint.\n";
        assert(0);
      }
    }
    data.named_scalars[name] = this;
  }

  const Scalar& evaluate() const {
    if (!is_dirty_) {
      return cache_;
    }
    if (blueprint_id_ < 0) {
      is_dirty_ = false;
      cache_ = value_;
      return value_;
    }
    Scalar net_output = evaluate_network_(blueprint_id_, output_id_);
    cache_ = is_residual ? value_ + net_output : net_output;
    is_dirty_ = false;
    return cache_;
  }

  /**
   * Defines a neural network connection for a scalar with a given name.
   * Once this scalar is registered with this name using assign(), the
   * specified input connections are made and neural network with the given
   * weights and biases is set up for this scalar.
   */
  static bool add_blueprint(const std::string& scalar_name,
                            const std::vector<std::string>& input_names,
                            const NeuralNetworkType& net) {
    return add_blueprint(std::vector<std::string>{scalar_name}, input_names,
                         net);
  }

  static bool add_blueprint(const std::vector<std::string>& output_names,
                            const std::vector<std::string>& input_names,
                            const NeuralNetworkType& net) {
    auto& data = get_data_();
    NeuralBlueprint blueprint{input_names, output_names, net};
    for (const auto& name : output_names) {
      if (data.scalar_to_blueprint.find(name) !=
          data.scalar_to_blueprint.end()) {
        std::cerr
            << "Error: duplicate definition of blueprint for output variable \""
            << name << "\".\n";
        assert(0);
        return false;
      }
      // register name with blueprint indices
      data.scalar_to_blueprint[name] = static_cast<int>(data.blueprints.size());
    }
    for (const auto& name : output_names) {
      if (data.scalar_to_blueprint.find(name) ==
          data.scalar_to_blueprint.end()) {
        data.scalar_input_to_blueprint[name] = {};
      }
      // register name with blueprint indices
      data.scalar_input_to_blueprint[name].push_back(
          static_cast<int>(data.blueprints.size()));
    }
    data.blueprints.push_back(blueprint);
    return true;
  }

  /**
   * Returns the number of total network parameters across all defined
   * blueprints.
   */
  static int num_blueprint_parameters() {
    int total = 0;
    const auto& blueprints = get_data_().blueprints;
    for (const auto& blueprint : blueprints) {
      total += blueprint.net.num_parameters();
    }
    return total;
  }

  static void set_blueprint_parameters(const std::vector<Scalar>& params) {
    int provided = static_cast<int>(params.size());
    int total = num_blueprint_parameters();
    if (provided != total) {
      fprintf(stderr,
              "Wrong number of blueprint parameters provided (%d). %d "
              "parameters are needed.\n",
              provided, total);
      assert(0);
      return;
    }
    int index = 0, next_index;
    auto& blueprints = get_data_().blueprints;
    for (auto& blueprint : blueprints) {
      int num_net = blueprint.net.num_parameters();
      next_index = index + num_net;
      std::vector<Scalar> net_params(params.begin() + index,
                                     params.begin() + next_index);
      blueprint.net.set_parameters(net_params);
#if DEBUG
      printf("Assigned %d parameters to network of scalar \"%s\".\n", num_net,
             entry.first.c_str());
#endif
      index = next_index;
    }
  }

  static void clear_all_blueprints() {
    auto& data = get_data_();
    data.clear();
  }

  /**
   * Save all blueprints as graphviz files with the given prefix.
   */
  static void save_graphviz(const std::string& filename_prefix = "") {
    for (const auto& blueprint : get_data_.blueprints) {
      std::string filename = filename_prefix;
      for (std::size_t i = 0; i < blueprint.input_names.size(); ++i) {
        filename += blueprint.input_names[i];
        if (i < blueprint.input_names.size() - 1) filename += "_";
      }
      filename += "--";
      for (std::size_t i = 0; i < blueprint.output_names.size(); ++i) {
        filename += blueprint.output_names[i];
        if (i < blueprint.output_names.size() - 1) filename += "_";
      }
      filename += ".dot";
      blueprint.net.save_graphviz(filename, blueprint.input_names,
                                  blueprint.output_names);
    }
  }

  /// Scalar operators create plain NeuralScalars that do not have neural
  /// networks.

  inline friend NeuralScalar operator+(const NeuralScalar& lhs,
                                       const NeuralScalar& rhs) {
    return NeuralScalar(lhs.evaluate() + rhs.evaluate());
  }

  inline friend NeuralScalar operator-(const NeuralScalar& lhs,
                                       const NeuralScalar& rhs) {
    return NeuralScalar(lhs.evaluate() - rhs.evaluate());
  }

  inline friend NeuralScalar operator*(const NeuralScalar& lhs,
                                       const NeuralScalar& rhs) {
    return NeuralScalar(rhs.evaluate() * lhs.evaluate());
  }
  inline friend NeuralScalar operator/(const NeuralScalar& lhs,
                                       const NeuralScalar& rhs) {
    return NeuralScalar(lhs.evaluate() / rhs.evaluate());
  }

  inline friend bool operator<(const NeuralScalar& lhs,
                               const NeuralScalar& rhs) {
    return lhs.evaluate() < rhs.evaluate();
  }
  inline friend bool operator<=(const NeuralScalar& lhs,
                                const NeuralScalar& rhs) {
    return lhs.evaluate() <= rhs.evaluate();
  }
  inline friend bool operator>(const NeuralScalar& lhs,
                               const NeuralScalar& rhs) {
    return lhs.evaluate() > rhs.evaluate();
  }

  inline friend bool operator==(const NeuralScalar& lhs,
                                const NeuralScalar& rhs) {
    return lhs.evaluate() == rhs.evaluate();
  }
  inline friend bool operator!=(const NeuralScalar& lhs,
                                const NeuralScalar& rhs) {
    return lhs.evaluate() != rhs.evaluate();
  }

  inline friend NeuralScalar operator-(const NeuralScalar& rhs) {
    return NeuralScalar(-rhs.evaluate());
  }

  inline NeuralScalar& operator+=(const NeuralScalar& lhs) {
    value_ += lhs.evaluate();
    is_dirty_ = true;
    return *this;
  }

  inline NeuralScalar& operator-=(const NeuralScalar& lhs) {
    value_ -= lhs.evaluate();
    is_dirty_ = true;
    return *this;
  }

  inline NeuralScalar& operator*=(const NeuralScalar& rhs) {
    value_ *= rhs.evaluate();
    is_dirty_ = true;
    return *this;
  }
  inline NeuralScalar& operator/=(const NeuralScalar& rhs) {
    value_ /= rhs.evaluate();
    is_dirty_ = true;
    return *this;
  }
};

template <typename Scalar, typename Utils>
struct NeuralScalarUtils {
  typedef ::NeuralScalar<Scalar, Utils> NeuralScalar;

  template <class T>
  static NeuralScalar fraction(T, T) = delete;  // C++11

  static NeuralScalar fraction(int num, int denom) {
    return scalar_from_double(double(num) / double(denom));
  }

  static NeuralScalar cos1(const NeuralScalar& v) {
    return Utils::cos1(v.evaluate());
  }
  static NeuralScalar sin1(const NeuralScalar& v) {
    return Utils::sin1(v.evaluate());
  }
  static NeuralScalar sqrt1(const NeuralScalar& v) {
    return Utils::sqrt1(v.evaluate());
  }
  static NeuralScalar atan2(const NeuralScalar& dy, const NeuralScalar& dx) {
    return Utils::atan2(dy.evaluate(), dx.evaluate());
  }
  static NeuralScalar asin(const NeuralScalar& v) {
    return Utils::asin(v.evaluate());
  }
  static NeuralScalar copysign(const NeuralScalar& x, const NeuralScalar& y) {
    return Utils::copysign(x.evaluate(), y.evaluate());
  }
  static NeuralScalar abs(const NeuralScalar& v) {
    return Utils::abs(v.evaluate());
  }
  static NeuralScalar pow(const NeuralScalar& a, const NeuralScalar& b) {
    return Utils::pow(a.evaluate(), b.evaluate());
  }
  static NeuralScalar exp(const NeuralScalar& v) {
    return Utils::exp(v.evaluate());
  }
  static NeuralScalar log(const NeuralScalar& v) {
    return Utils::log(v.evaluate());
  }
  static NeuralScalar tanh(const NeuralScalar& v) {
    return Utils::tanh(v.evaluate());
  }
  static NeuralScalar min1(const NeuralScalar& a, const NeuralScalar& b) {
    return Utils::min1(a.evaluate(), b.evaluate());
  }
  static NeuralScalar max1(const NeuralScalar& a, const NeuralScalar& b) {
    return Utils::max1(a.evaluate(), b.evaluate());
  }

  static NeuralScalar zero() { return scalar_from_double(0.); }
  static NeuralScalar one() { return scalar_from_double(1.); }

  static NeuralScalar two() { return scalar_from_double(2.); }
  static NeuralScalar half() { return scalar_from_double(0.5); }
  static NeuralScalar pi() { return scalar_from_double(M_PI); }
  static NeuralScalar half_pi() { return scalar_from_double(M_PI / 2.); }

  static double getDouble(const NeuralScalar& v) {
    return Utils::getDouble(v.evaluate());
  }
  static inline NeuralScalar scalar_from_double(double value) {
    return NeuralScalar(Scalar(value));
  }

  static NeuralScalar scalar_from_string(const std::string& txt) {
    return NeuralScalar(Utils::scalar_from_string(txt));
  }

  template <class T>
  static NeuralScalar convert(T) = delete;  // C++11
  static NeuralScalar convert(int value) {
    return scalar_from_double((double)value);
  }

  static void FullAssert(bool a) {
    if (!a) {
      printf("!");
      assert(0);
      exit(0);
    }
  }

  static inline std::vector<NeuralScalar> to_neural(
      const std::vector<Scalar>& values) {
    std::vector<NeuralScalar> output(values.begin(), values.end());
    return output;
  }
  static inline std::vector<Scalar> from_neural(
      const std::vector<NeuralScalar>& values) {
    std::vector<Scalar> output(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
      output[i] = values[i].evaluate();
    }
    return output;
  }
};

template <typename Scalar, typename Utils>
struct is_neural_scalar {
  static constexpr bool value = false;
};
template <typename Scalar, typename Utils>
struct is_neural_scalar<NeuralScalar<Scalar, Utils>,
                        NeuralScalarUtils<Scalar, Utils>> {
  static constexpr bool value = true;
};

#define NEURAL_ASSIGN(var, name)                                      \
  if constexpr (is_neural_scalar<TinyScalar, TinyConstants>::value) { \
    var.assign(name);                                                 \
  }

#endif  // NEURAL_SCALAR_H