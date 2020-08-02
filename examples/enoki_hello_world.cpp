#include <enoki/autodiff.h>
#include <enoki/cuda.h>
#include <enoki/quaternion.h>
#include <enoki/special.h>  // for erf()

#define TINY_INLINE ENOKI_INLINE

struct EnokiAlgebra {
  using Scalar = double;
  using Vector3 = enoki::Array<Scalar, 3>;
  using Vector6 = enoki::Array<Scalar, 6>;
  using Matrix3 = enoki::Matrix<Scalar, 3>;
  using Matrix6 = enoki::Matrix<Scalar, 6>;
  using Quaternion = enoki::Quaternion<Scalar>;

  template <typename T>
  ENOKI_INLINE static auto transpose(const T &matrix) {
    return enoki::transpose(matrix);
  }

  template <typename T1, typename T2>
  ENOKI_INLINE static auto cross(const T1 &vector_a, const T2 &vector_b) {
    return enoki::cross(vector_a, vector_b);
  }

  /**
   * Cross product in matrix form.
   */
  ENOKI_INLINE static Matrix3 cross_matrix(const Vector3 &v) {
    return Matrix3(0., -v[2], v[1], v[2], 0., -v[0], -v[1], v[0], 0.);
  }

  ENOKI_INLINE static Matrix3 diagonal3(const Vector3 &v) {
    return Matrix3(v[0], 0, 0, 0, v[1], 0, 0, 0, v[2]);
  }
  ENOKI_INLINE static Matrix3 diagonal3(const Scalar &v) { return Matrix3(v); }

  ENOKI_INLINE static Scalar zero() { return 0; }
  ENOKI_INLINE static Scalar one() { return 1; }

  ENOKI_INLINE static Vector3 zero3() { return Vector3(0); }
  ENOKI_INLINE static Vector3 unit3_x() { return Vector3(1, 0, 0); }
  ENOKI_INLINE static Vector3 unit3_y() { return Vector3(0, 1, 0); }
  ENOKI_INLINE static Vector3 unit3_z() { return Vector3(0, 0, 1); }

  template <std::size_t Size1, std::size_t Size2>
  ENOKI_INLINE static void assign_block(
      enoki::Matrix<Scalar, Size1> &output,
      const enoki::Matrix<Scalar, Size2> &input, std::size_t i, std::size_t j,
      std::size_t m = Size2, std::size_t n = Size2, std::size_t input_i = 0,
      std::size_t input_j = 0) {
    assert(i + m <= Size1 && j + n <= Size1);
    assert(input_i + m <= Size2 && input_j + n <= Size2);
    for (std::size_t ii = 0; ii < m; ++ii) {
      for (std::size_t jj = 0; jj < n; ++jj) {
        output(ii + i, jj + j) = input(ii + input_i, jj + input_j);
      }
    }
  }

  ENOKI_INLINE static Matrix3 quat_to_matrix(const Quaternion &quat) {
    return enoki::quat_to_matrix<Matrix3>(quat);
  }
  ENOKI_INLINE static Matrix3 quat_to_matrix(const Scalar &x, const Scalar &y,
                                             const Scalar &z, const Scalar &w) {
    return enoki::quat_to_matrix<Matrix3>(Quaternion(x, y, z, w));
  }
  ENOKI_INLINE static Quaternion matrix_to_quat(const Matrix3 &m) {
    return enoki::matrix_to_quat(m);
  }
  ENOKI_INLINE static Quaternion axis_angle_quaternion(const Vector3 &axis,
                                                       const Scalar &angle) {
    return enoki::rotate<Quaternion, Vector3>(axis, angle);
  }
  ENOKI_INLINE static Matrix3 rotation_x_matrix(const Scalar &angle) {
    Scalar c = enoki::cos(angle);
    Scalar s = enoki::sin(angle);
    return Matrix3(1, 0, 0, 0, c, -s, 0, s, c);
  }

  ENOKI_INLINE static Matrix3 rotation_y_matrix(const Scalar &angle) {
    Scalar c = enoki::cos(angle);
    Scalar s = enoki::sin(angle);
    return Matrix3(c, 0, s, 0, 1, 0, -s, 0, c);
  }

  ENOKI_INLINE static Matrix3 rotation_z_matrix(const Scalar &angle) {
    Scalar c = enoki::cos(angle);
    Scalar s = enoki::sin(angle);
    return Matrix3(c, -s, 0, s, c, 0, 0, 0, 1);
  }

  template <std::size_t Size>
  ENOKI_INLINE static void set_zero(enoki::Matrix<Scalar, Size> &m) {
    m = 0;
  }
  template <std::size_t Size>
  ENOKI_INLINE static void set_zero(enoki::Array<Scalar, Size> &m) {
    m = 0;
  }

  ENOKI_INLINE static double to_double(const Scalar &s) {
    return static_cast<double>(s);
  }

  template <typename T>
  static void print(const std::string &title, T object) {
    std::cout << title << "\n" << object << std::endl;
  }

  EnokiAlgebra() = delete;
};

template <typename Algebra>
struct SpatialVector {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;

  Vector3 top{0.};
  Vector3 bottom{0.};

  TINY_INLINE Scalar &operator[](int i) {
    if (i < 3)
      return top[i];
    else
      return bottom[i - 3];
  }
  const TINY_INLINE Scalar &operator[](int i) const {
    if (i < 3)
      return top[i];
    else
      return bottom[i - 3];
  }

  ENOKI_STRUCT(SpatialVector, top, bottom)
};
ENOKI_STRUCT_SUPPORT(SpatialVector, top, bottom)

template <typename Algebra>
struct Transform {
  using Vector3 = typename Algebra::Vector3;
  using Matrix3 = typename Algebra::Matrix3;

  Vector3 translation{0.};
  Matrix3 rotation{Matrix3(1.)};

  friend std::ostream &operator<<(std::ostream &os,
                                  const Transform<Algebra> &tf) {
    os << "[ translation: " << tf.translation << "  rotation: " << tf.rotation
       << " ]";
    return os;
  }

  void set_identity() {
    // set all vector entries to zero
    translation = 0.;
    // set diagonal entries to one, others to zero
    rotation = Matrix3(1.);
  }

  /**
   * X1*X2 = plx(E1*E2, r2 + E2T*r1)
   */
  Transform operator*(const Transform &t) const {
    Transform tr = *this;
    tr.translation += rotation * t.translation;
    tr.rotation *= t.rotation;
    return tr;
  }

  Vector3 apply(const Vector3 &point) const {
    return rotation * point + translation;
  }
  Vector3 apply_inverse(const Vector3 &point) const {
    return Algebra::transpose(rotation) * (point - translation);
  }

  Transform get_inverse() const {
    Transform inv;
    inv.rotation = Algebra::transpose(rotation);
    inv.translation = inv.rotation * -translation;
    return inv;
  }

  /**
   * V = mv(w, v)
   * X*V = mv(E*w, E*(v - r x w))
   */
  template <typename SpatialVector>
  SpatialVector apply(const SpatialVector &inVec) const {
    SpatialVector outVec;

    Vector3 rxw = inVec.Algebra::cross(inVec.top, translation);
    Vector3 v_rxw = inVec.bottom + rxw;

    Vector3 tmp3 = Algebra::transpose(rotation) * v_rxw;
    Vector3 tmp4 = Algebra::transpose(rotation) * inVec.top;

    outVec.top = tmp4;
    outVec.bottom = tmp3;

    return outVec;
  }

  /**
   * V = mv(w, v)
   * inv(X)*V = mv(ET*w, ET*v + r x (ET*w))
   */
  template <typename SpatialVector>
  SpatialVector apply_inverse(const SpatialVector &inVec) const {
    SpatialVector outVec;
    outVec.top = rotation * inVec.top;
    outVec.bottom =
        rotation * inVec.bottom + Algebra::cross(translation, outVec.top);
    return outVec;
  }

  /**
   * F = fv(n, f)
   * XT*F = fv(ETn + rxETf, ETf)
   */
  template <typename SpatialVector>
  SpatialVector apply_transpose(const SpatialVector &inVec) const {
    SpatialVector outVec;
    outVec.bottom = rotation * inVec.bottom;
    outVec.top = rotation * inVec.top;
    outVec.top += cross_matrix(translation) * outVec.bottom;

    return outVec;
  }

  /**
   * F = fv(n, f)
   * X^* F = fv(E(n - rxf), Ef)
   */
  template <typename SpatialVector>
  SpatialVector apply_inverse_transpose(const SpatialVector &inVec) const {
    const Vector3 &n = inVec.top;
    const Vector3 &f = inVec.bottom;
    Matrix3 Et = Algebra::transpose(rotation);
    SpatialVector outVec;
    outVec.top = Et * (n - Algebra::cross(translation, f));
    outVec.bottom = Et * f;
    return outVec;
  }

  ENOKI_STRUCT(Transform, translation, rotation)
};
ENOKI_STRUCT_SUPPORT(Transform, translation, rotation)

enum JointType {
  JOINT_FIXED = -1,
  JOINT_PRISMATIC_X = 0,
  JOINT_PRISMATIC_Y,
  JOINT_PRISMATIC_Z,
  JOINT_PRISMATIC_AXIS,
  JOINT_REVOLUTE_X,
  JOINT_REVOLUTE_Y,
  JOINT_REVOLUTE_Z,
  JOINT_REVOLUTE_AXIS,
  JOINT_INVALID,
};

template <typename Algebra>
struct Inertia {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using Matrix3 = typename Algebra::Matrix3;
  using Matrix6 = typename Algebra::Matrix6;
  typedef ::Transform<Algebra> Transform;
  typedef ::SpatialVector<Algebra> SpatialVector;

  /**
   * Mass \f$m\f$.
   */
  Scalar mass{0};

  /**
   * Center of mass, also denoted as \f$h\f$.
   */
  Vector3 com{0};

  Matrix3 inertia{Algebra::diagonal3(1)};

  Inertia(const Inertia<Algebra> &rbi) = default;

  Inertia(const Scalar &mass, const Scalar &Ixx, const Scalar &Iyx,
          const Scalar &Iyy, const Scalar &Izx, const Scalar &Izy,
          const Scalar &Izz)
      : mass(mass) {
    inertia = {{Ixx, Iyx, Izx}, {Iyx, Iyy, Izy}, {Izx, Izy, Izz}};
  }

  Inertia(const Scalar &mass, const Vector3 &com, const Matrix3 &inertia)
      : mass(mass), com(com), inertia(inertia) {}

  Inertia(const Scalar &mass, const Vector3 &com, const Scalar &Ixx,
          const Scalar &Iyx, const Scalar &Iyy, const Scalar &Izx,
          const Scalar &Izy, const Scalar &Izz)
      : mass(mass), com(com) {
    inertia = {{Ixx, Iyx, Izx}, {Iyx, Iyy, Izy}, {Izx, Izy, Izz}};
  }

  Inertia(const Matrix6 &m)
      : mass(m(3, 3)),
        com{-m(1, 5), m(0, 5), -m(0, 4)},
        inertia(m(0, 0), m(1, 0), m(2, 0), m(0, 1), m(1, 1), m(2, 1), m(0, 2),
                m(1, 2), m(2, 2)) {}

  void set_zero() {
    mass = 0;
    Algebra::set_zero(com);
    Algebra::set_zero(inertia);
  }

  /**
   * Represents spatial inertia matrix where the inertia coincide with the given
   * center of mass.
   * @param mass The mass.
   * @param com The center of mass \f$C\f$.
   * @param inertiaC Inertia matrix \f$I_C\f$ at the center of mass \f$C\f$.
   * @return The spatial inertia data structure.
   */
  static Inertia from_mass_com_i(const Scalar &mass, const Vector3 &com,
                                 const Matrix3 &inertiaC) {
    const Matrix3 crossCom = Algebra::cross_matrix(com);
    const Matrix3 I = inertiaC + crossCom * Algebra::transpose(crossCom) * mass;
    return Inertia(mass, com * mass, I(0, 0), I(1, 0), I(1, 1), I(2, 0),
                   I(2, 1), I(2, 2));
  }

  /**
   * Represents spatial inertia matrix where the inertia coincide with the given
   * center of mass.
   * @param mass The mass.
   * @param com The center of mass \f$C\f$.
   * @param gyrationRadii Radii of gyration at the center of mass (diagonal of
   * inertia matrix).
   * @return The spatial inertia data structure.
   */
  static Inertia from_mass_com_xyz(const Scalar &mass, const Vector3 &com,
                                   const Vector3 &gyrationRadii) {
    return Inertia(mass, com * mass, gyrationRadii(0), 0, gyrationRadii(1), 0,
                   0, gyrationRadii(2));
  }

  Matrix6 matrix() const {
    Matrix6 m;
    Algebra::assign_block(m, inertia, 0, 0);
    const auto crossCom = Algebra::cross_matrix(com);
    Algebra::assign_block(m, crossCom, 0, 3);
    Algebra::assign_block(m, -crossCom, 3, 0);
    Algebra::assign_block(m, Algebra::diagonal3(mass), 3, 3);

    return m;
  }

  SpatialVector operator*(const SpatialVector &v) const {
    SpatialVector result;
    result.top = inertia * v.top + Algebra::cross(com, v.bottom);
    result.bottom = mass * v.bottom - Algebra::cross(com, v.top);
    return result;
  }

  Inertia operator+(const Inertia &rbi) const {
    return Inertia(mass + rbi.mass, com + rbi.com, inertia + rbi.inertia);
  }

  Inertia &operator+=(const Inertia &rbi) {
    mass += rbi.mass;
    com += rbi.com;
    inertia += rbi.inertia;
    return *this;
  }

  Inertia &operator=(const Matrix6 &m) {
    mass = m(3, 3);
    com = Vector3(-m(1, 5), m(0, 5), -m(0, 4));
    Algebra::assign_block(inertia, m, 0, 0, 3, 3);
    return *this;
  }

  /**
   * Computes \f$ X^* I X^{-1} \f$.
   */
  void transform(const Transform &transform) {
    const Matrix3 rx = Algebra::cross_matrix(transform.translation);
    // E(I + rx hx + (h - mr)x rx) E^T
    inertia = transform.rotation *
              (inertia + rx * Algebra::cross_matrix(com) +
               Algebra::cross_matrix(com - mass * transform.translation) * rx) *
              Algebra::transpose(transform.rotation);
    // E(h - mr)
    com = transform.rotation * (com - mass * transform.translation);
  }

  /**
   * Computes \f$ X^T I X \f$.
   */
  void transform_transpose(const Transform &transform) {
    // E^T h + mr
    const Matrix3 E_T = Algebra::transpose(transform.rotation);
    const Vector3 E_T_mr = E_T * com + mass * transform.translation;
    const Matrix3 crossTranslation =
        Algebra::cross_matrix(transform.translation);
    // E^T I E - rx(E^T h)x - (E^T h + mr)x rx
    inertia = (E_T * inertia * transform.rotation -
               crossTranslation * Algebra::cross_matrix(E_T * com) -
               Algebra::cross_matrix(E_T_mr) * crossTranslation);
    // E^T h + mr
    com = E_T_mr;
  }

  ENOKI_STRUCT(Inertia, mass, com, inertia)
};
ENOKI_STRUCT_SUPPORT(Inertia, mass, com, inertia)

template <typename Algebra>
class Link {
  typedef ::Transform<Algebra> Transform;
  typedef ::SpatialVector<Algebra> SpatialVector;
  typedef ::Inertia<Algebra> Inertia;
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using Matrix3 = typename Algebra::Matrix3;

 public:
  Link() = default;
  Link(JointType joint_type, Transform &parent_link_to_joint,
       const Inertia &inertia)
      : joint_type(joint_type), X_T(parent_link_to_joint), I(inertia) {}

  Transform X_T;       // parent_link_to_joint
  Transform X_J;       // joint_to_child_link    //depends on q
  Transform X_parent;  // parent_link_to_child_link

  JointType joint_type{JOINT_REVOLUTE_Z};

  Transform X_world;  // world_to_link
  SpatialVector vJ;   // local joint velocity (relative to parent link)
  SpatialVector v;    // global joint velocity (relative to world)
  SpatialVector a;    // acceleration (relative to world)
  SpatialVector c;    // velocity product acceleration

  Inertia I;   // local spatial inertia (constant) // TODO replace
               // by its original terms (COM, gyration etc.)
  Inertia IA;  // spatial articulated inertia, IC in CRBA

  SpatialVector pA;  // bias forces or zero-acceleration forces
  SpatialVector S;   // motion subspace (spatial joint axis/matrix)

  SpatialVector U;  // temp var in ABA, page 130
  Scalar d;         // temp var in ABA, page 130
  Scalar u;         // temp var in ABA, page 130
  SpatialVector f;  // temp var in RNEA, page 183

  SpatialVector f_ext;  // user-defined external force in world frame

  // These two variables are managed by MultiBody and should not be changed.
  int parent_index{-1};  // index of parent link in MultiBody
  int index{-1};         // index of this link in MultiBody

  // std::vector<const Geometry<Scalar, Constants> *> collision_geometries;
  // std::vector<Transform> X_collisions;  // offset of collision geometries
  // (relative to this link frame)
  std::vector<int> visual_uids1;
  std::vector<int> visual_uids2;
  std::vector<Transform>
      X_visuals;  // offset of geometry (relative to this link frame)
  std::string link_name;
  std::string joint_name;
  // index in MultiBody q / qd arrays
  int q_index{-2};
  int qd_index{-2};

  Scalar stiffness{0};
  Scalar damping{0};

  void set_joint_type(JointType type,
                      const Vector3 &axis = Algebra::unit3_x()) {
    joint_type = type;
    S.set_zero();
    switch (joint_type) {
      case JOINT_PRISMATIC_X:
        S.bottom[0] = 1.;
        break;
      case JOINT_PRISMATIC_Y:
        S.bottom[1] = 1.;
        break;
      case JOINT_PRISMATIC_Z:
        S.bottom[2] = 1.;
        break;
      case JOINT_PRISMATIC_AXIS:
        S.bottom = axis;
        break;
      case JOINT_REVOLUTE_X:
        S.top[0] = 1.;
        break;
      case JOINT_REVOLUTE_Y:
        S.top[1] = 1.;
        break;
      case JOINT_REVOLUTE_Z:
        S.top[2] = 1.;
        break;
      case JOINT_REVOLUTE_AXIS:
        S.top = axis;
        break;
      case JOINT_FIXED:
        break;
      default:
        fprintf(stderr,
                "Error: Unknown joint type encountered in " __FILE__ ":%i\n",
                __LINE__);
    }
  }

  void jcalc(const Scalar &q, Transform *X_J, Transform *X_parent) const {
    X_J->set_identity();
    X_parent->set_identity();
    switch (joint_type) {
      case JOINT_PRISMATIC_X:
        X_J->translation[0] = q;
        break;
      case JOINT_PRISMATIC_Y:
        X_J->translation[1] = q;
        break;
      case JOINT_PRISMATIC_Z:
        X_J->translation[2] = q;
        break;
      case JOINT_PRISMATIC_AXIS: {
        const Vector3 &axis = S.bottom;
        X_J->translation = axis * q;
        break;
      }
      case JOINT_REVOLUTE_X:
        X_J->rotation.set_rotation_x(q);
        break;
      case JOINT_REVOLUTE_Y:
        X_J->rotation.set_rotation_y(q);
        break;
      case JOINT_REVOLUTE_Z:
        X_J->rotation.set_rotation_z(q);
        break;
      case JOINT_REVOLUTE_AXIS: {
        const Vector3 &axis = S.bottom;
        const auto quat = Algebra::axis_angle_quaternion(axis, q);
        X_J->rotation.rotation = Algebra::quat_to_matrix(quat);
        break;
      }
      case JOINT_FIXED:
        // Transform is set to identity in its constructor already
        // and never changes.
        break;
      default:
        fprintf(stderr,
                "Error: Unknown joint type encountered in " __FILE__ ":%i\n",
                __LINE__);
    }
    *X_parent = X_T * (*X_J);
  }

  inline void jcalc(const Scalar &qd, SpatialVector *v_J) const {
    switch (joint_type) {
      case JOINT_PRISMATIC_X:
        v_J->bottom[0] = qd;
        break;
      case JOINT_PRISMATIC_Y:
        v_J->bottom[1] = qd;
        break;
      case JOINT_PRISMATIC_Z:
        v_J->bottom[2] = qd;
        break;
      case JOINT_PRISMATIC_AXIS: {
        const Vector3 &axis = S.bottom;
        v_J->bottom = axis * qd;
        break;
      }
      case JOINT_REVOLUTE_X:
        v_J->top[0] = qd;
        break;
      case JOINT_REVOLUTE_Y:
        v_J->top[1] = qd;
        break;
      case JOINT_REVOLUTE_Z:
        v_J->top[2] = qd;
        break;
      case JOINT_REVOLUTE_AXIS: {
        const Vector3 &axis = S.top;
        v_J->top = axis * qd;
        break;
      }
      case JOINT_FIXED:
        break;
      default:
        fprintf(stderr,
                "Error: Unknown joint type encountered in " __FILE__ ":%i\n",
                __LINE__);
    }
  }

  inline void jcalc(const Scalar &q) { jcalc(q, &X_J, &X_parent); }

  inline void jcalc(const Scalar &q, const Scalar &qd) {
    jcalc(q);
    jcalc(qd, &vJ);
  }
};

template <typename Algebra>
class MultiBody {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using Matrix3 = typename Algebra::Matrix3;
  using Matrix6 = typename Algebra::Matrix6;
  typedef ::Transform<Algebra> Transform;
  typedef ::SpatialVector<Algebra> SpatialVector;
  typedef ::Link<Algebra> Link;
  typedef ::Inertia<Algebra> Inertia;

  /**
   * Number of degrees of freedom, excluding floating-base coordinates.
   */
  int dof_{0};

 public:
  std::vector<Link> links;

  /**
   * Dimensionality of joint positions q (including 7-DoF floating-base
   * coordinates if this system is floating-base).
   */
  TINY_INLINE int dof() const { return is_floating ? dof_ + 7 : dof_; }
  /**
   * Dimensionality of joint velocities qd and accelerations qdd (including
   * 6-DoF base velocity and acceleration, if this system is floating-base).
   */
  TINY_INLINE int dof_qd() const { return is_floating ? dof_ + 6 : dof_; }

  /**
   * Indices in `tau` that are controllable, i.e. actuated.
   * For floating-base system, the index 0 corresponds to the first degree of
   * freedom not part of the 6D floating-base coordinates.
   */
  std::vector<int> control_indices;
  /**
   * Dimensionality of control input, i.e. number of actuated DOFs.
   */
  TINY_INLINE int dof_actuated() const {
    return static_cast<int>(control_indices.size());
  }

  /**
   * Whether this system is floating or fixed to the world frame.
   */
  bool is_floating{false};

  // quantities related to floating base
  SpatialVector base_velocity;       // v_0
  SpatialVector base_acceleration;   // a_0
  SpatialVector base_applied_force;  // f_ext_0 in world frame
  SpatialVector base_force;          // f_0 (used by RNEA)
  SpatialVector base_bias_force;     // pA_0
  Inertia base_inertia;              // I_0
  Inertia base_articulated_inertia;  // IA_0
  Transform base_X_world;

  std::vector<int> visual_uids1;
  std::vector<int> visual_uids2;
  // offset of geometry (relative to the base frame)
  std::vector<Transform> X_visuals;

  // std::vector<const TinyGeometry<Scalar, Algebra> *>
  //     collision_geometries;
  // offset of collision geometries (relative to this link frame)
  std::vector<Transform> X_collisions;

  std::vector<Scalar> q, qd, qdd, tau;

  explicit MultiBody(bool isFloating = false) : is_floating(isFloating) {}

  /**
   * Set 3D base position in world coordinates.
   */
  void set_position(const Vector3 &initial_position) {
    base_X_world.translation = initial_position;
    if (is_floating) {
      q[4] = initial_position[0];
      q[5] = initial_position[1];
      q[6] = initial_position[2];
    }
  }

  /**
   * Ensures that the joint coordinates q, qd, qdd, tau are initialized
   * properly in the MultiBody member variables.
   */
  void initialize() {
    // make sure dof and the q / qd indices in the links are accurate
    int q_index = is_floating ? 7 : 0;
    int qd_index = is_floating ? 6 : 0;
    dof_ = 0;  // excludes floating-base DOF
    for (Link &link : links) {
      assert(link.index >= 0);
      link.q_index = q_index;
      link.qd_index = qd_index;
      if (link.joint_type != JOINT_FIXED) {
        ++q_index;
        ++qd_index;
        ++dof_;
      } else {
        link.q_index = -2;
        link.qd_index = -2;
      }
    }

    if (static_cast<int>(q.size()) != dof()) {
      q.resize(dof(), Algebra::zero());
    }
    for (Scalar &v : q) {
      v = Algebra::zero();
    }
    if (is_floating) {
      q[3] = Algebra::one();  // make sure orientation is valid
    }
    if (static_cast<int>(qd.size()) != dof_qd()) {
      qd.resize(dof_qd(), Algebra::zero());
    }
    for (Scalar &v : qd) {
      v = Algebra::zero();
    }
    if (static_cast<int>(qdd.size()) != dof_qd()) {
      qdd.resize(dof_qd(), Algebra::zero());
    }
    for (Scalar &v : qdd) {
      v = Algebra::zero();
    }
    if (static_cast<int>(tau.size()) != dof_actuated()) {
      tau.resize(dof_actuated(), Algebra::zero());
    }
    for (Scalar &v : tau) {
      v = Algebra::zero();
    }

    // (Re-)create actuator to make sure it has the right degrees of freedom.
    // if (actuator) {
    //   delete actuator;
    //   actuator = new TinyActuator(dof_actuated());
    // }
  }

  /**
   * Copy constructor. Skips visualization members, temporary variables.
   * The actuator is not copied, but the original pointer `m_actuator` is
   * carried over.
   */
  template <typename Algebra2>
  MultiBody(const MultiBody<Algebra2> &mb)
      : links(mb.links),
        dof_(mb.dof_),
        // actuator(mb.actuator),
        control_indices(mb.control_indices),
        is_floating(mb.is_floating),
        base_velocity(mb.base_velocity),
        base_acceleration(mb.base_acceleration),
        base_applied_force(mb.base_applied_force),
        base_force(mb.base_force),
        base_bias_force(mb.base_bias_force),
        base_inertia(mb.base_inertia),
        base_X_world(mb.base_X_world),
        // collision_geometries(mb.collision_geometries),
        X_collisions(mb.X_collisions),
        q(mb.q),
        qd(mb.qd),
        qdd(mb.qdd),
        tau(mb.tau) {}

  virtual ~MultiBody() {
    // if (m_actuator) {
    //   delete actuator;
    // }
  }

  void print_state() const {
    printf("q: [");
    for (int i = 0; i < dof(); ++i) {
      if (i > 0) printf(" ");
      printf("%.2f", Algebra::to_double(q[i]));
    }
    printf("] \tqd: [");
    for (int i = 0; i < dof_qd(); ++i) {
      if (i > 0) printf(" ");
      printf("%.2f", Algebra::to_double(qd[i]));
    }
    printf("] \tqdd: [");
    for (int i = 0; i < dof_qd(); ++i) {
      if (i > 0) printf(" ");
      printf("%.2f", Algebra::to_double(qdd[i]));
    }
    printf("] \ttau: [");
    for (int i = 0; i < dof_actuated(); ++i) {
      if (i > 0) printf(" ");
      printf("%.2f", Algebra::to_double(tau[i]));
    }
    printf("]\n");
  }

  const Transform &get_world_transform(int link) const {
    if (link == -1) {
      return base_X_world;
    } else {
      return links[link].X_world;
    }
  }

  /**
   * Compute center of mass of link in world coordinates.
   * @param link Index of link in `links`.
   * @return 3D coordinates of center of mass in world coordinates.
   */
  const Vector3 get_world_com(int link) const {
    const Transform &tf = get_world_transform(link);
    if (link == -1) {
      return tf.apply(base_inertia.com);
    } else {
      return tf.apply(links[link].I.com);
    }
  }

  inline Scalar get_q_for_link(const std::vector<Scalar> &q,
                               int link_index) const {
    if (q.empty()) return Algebra::zero();
    const Link &link = links[link_index];
    return link.joint_type == JOINT_FIXED ? Algebra::zero() : q[link.q_index];
  }
  inline Scalar get_q_for_link(int link_index) const {
    get_q_for_link(q, link_index);
  }

  inline Scalar get_qd_for_link(const std::vector<Scalar> &qd,
                                int link_index) const {
    if (qd.empty()) return Algebra::zero();
    const Link &link = links[link_index];
    return link.joint_type == JOINT_FIXED ? Algebra::zero() : qd[link.qd_index];
  }
  inline Scalar get_qd_for_link(int link_index) const {
    return get_qd_for_link(qd, link_index);
  }

  inline Scalar get_qdd_for_link(const std::vector<Scalar> &qdd,
                                 int link_index) const {
    return get_qd_for_link(qdd, link_index);
  }
  inline Scalar get_qdd_for_link(int link_index) const {
    return get_qdd_for_link(qdd, link_index);
  }

  inline Scalar get_tau_for_link(const std::vector<Scalar> &tau,
                                 int link_index) const {
    if (tau.empty()) return Algebra::zero();
    const Link &link = links[link_index];
    int offset = is_floating ? -6 : 0;
    return link.joint_type == JOINT_FIXED ? Algebra::zero()
                                          : tau[link.qd_index + offset];
  }
  inline Scalar get_tau_for_link(int link_index) const {
    return get_tau_for_link(tau, link_index);
  }

  /**
   * Set joint torques and external forces in all links and the base to zero.
   */
  void clear_forces() {
    base_applied_force.set_zero();
    for (Link &link : links) {
      link.f_ext.set_zero();
    }
    for (int i = 0; i < dof_actuated(); ++i) {
      tau[i] = Algebra::zero();
    }
  }

  /**
   * Implements the first phase in ABA, CRBA and RNEA, that computes the
   * joint and body transforms, velocities and bias forces.
   * Initializes articulated inertia with the local body inertia.
   *
   * Joint positions q must have dimension of dof().
   * Joint velocities qd must have dimension of dof_qd().
   * If no joint velocities qd are given, qd is assumed to be zero.
   * If no joint accelerations qdd are given, qdd is assumed to be zero.
   */
  void forward_kinematics(
      const std::vector<Scalar> &q,
      const std::vector<Scalar> &qd = std::vector<Scalar>(),
      const std::vector<Scalar> &qdd = std::vector<Scalar>()) {
    assert(q.size() == dof());
    assert(qd.empty() || qd.size() == dof_qd());
    assert(qdd.empty() || qdd.size() == dof_qd());

    if (is_floating) {
      // update base-world transform from q, and update base velocity from qd
      base_X_world.rotation = Algebra::quat_to_matrix(q[0], q[1], q[2], q[3]);
      base_X_world.translation = Vector3(q[4], q[5], q[6]);
      if (!qd.empty()) {
        base_velocity.top = Vector3(qd[0], qd[1], qd[2]);
        base_velocity.bottom = Vector3(qd[3], qd[4], qd[5]);
      } else {
        Algebra::set_zero(base_velocity);
      }

      SpatialVector I0_mul_v0 = base_inertia.mul_org(base_velocity);
      base_bias_force = base_velocity.crossf(I0_mul_v0) - base_applied_force;

      base_articulated_inertia = base_inertia;
    }

    for (int i = 0; i < links.size(); i++) {
      Link &link = links[i];
      int parent = link.parent_index;

      // update joint transforms, joint velocity (if available)
      Scalar q_val = get_q_for_link(q, i);
      Scalar qd_val = get_qd_for_link(qd, i);
      link.jcalc(q_val, qd_val);

      if (parent >= 0 || is_floating) {
        const Transform &parent_X_world =
            parent >= 0 ? links[parent].X_world : base_X_world;
        link.X_world = parent_X_world * link.X_parent;
        const SpatialVector &parentVelocity =
            parent >= 0 ? links[parent].v : base_velocity;
        SpatialVector xv = link.X_parent.apply(parentVelocity);
        link.v = xv + link.vJ;
      } else {
        link.X_world = base_X_world * link.X_parent;
        link.v = link.vJ;
      }
      SpatialVector v_x_vJ = link.v.crossm(link.vJ);
      link.c = v_x_vJ /*+link.c_J[i]*/;

      link.IA = link.I;
      SpatialVector I_mul_v = link.I.mul_inv(link.v);
      SpatialVector f_ext = link.X_world.apply_inverse_transpose(link.f_ext);

      // #ifdef NEURAL_SIM
      //       if (i >= 3) {
      //         if constexpr (is_neural_scalar<Scalar, Algebra>::value) {
      //           // Inputs: Position.
      //           link.X_world.translation[0].assign("link/pos/x");
      //           link.X_world.translation[1].assign("link/pos/y");
      //           Scalar link_pos_yaw = Algebra::atan2(
      //               link.X_world.rotation(0, 1), link.X_world.rotation(0,
      //               0));
      //           link_pos_yaw.assign("link/pos/yaw");

      //           // Inputs: Velocity.
      //           link.v[3].assign("link/vel/x");
      //           link.v[4].assign("link/vel/y");
      //           link.v[2].assign("link/vel/yaw");

      //           // Outputs: Applied Force.
      //           f_ext[3].assign("link/external_force/x");
      //           f_ext[4].assign("link/external_force/y");
      //           f_ext[2].assign("link/external_force/yaw");

      //           // Cache the outputs.
      //           f_ext[3].evaluate();
      //           f_ext[4].evaluate();
      //           f_ext[2].evaluate();
      //         }
      //       }
      // #endif

      link.pA = link.v.crossf(I_mul_v) - f_ext;
#ifdef DEBUG
      Algebra::print("link.IA", link.IA);
      Algebra::print("I_mul_v", I_mul_v);
      Algebra::print("link.pA", link.pA);
#endif
      // compute helper temporary variables for floating-base RNEA
      const SpatialVector &parent_a =
          parent >= 0 ? links[parent].a : base_acceleration;
      link.a = link.X_parent.apply(parent_a) + v_x_vJ;
      if (!qdd.empty()) {
        link.a += link.S * get_qdd_for_link(qdd, i);
      }
      link.f = link.I.mul_inv(link.a) + link.pA;
    }
  }

  /**
   * Updates the forward kinematics given the q, qd coordinates stored in this
   * model.
   */
  void forward_kinematics() { forward_kinematics(q, qd); }

  void forward_dynamics(const std::vector<Scalar> &q,
                        const std::vector<Scalar> &qd,
                        const std::vector<Scalar> &tau, const Vector3 &gravity,
                        std::vector<Scalar> &qdd) {
    assert(q.size() == dof());
    assert(qd.size() == dof_qd());
    assert(qdd.size() == dof_qd());
    assert(static_cast<int>(tau.size()) == m_dof);

    SpatialVector spatial_gravity(
        Vector3(Algebra::zero(), Algebra::zero(), Algebra::zero()), gravity);

#ifdef NEURAL_SIM
    for (int i = 0; i < dof(); ++i) {
      NEURAL_ASSIGN(q[i], "q_" + std::to_string(i));
    }
    for (int i = 0; i < dof_qd(); ++i) {
      NEURAL_ASSIGN(qd[i], "qd_" + std::to_string(i));
    }
#endif

    forward_kinematics(q, qd);

    for (int i = links.size() - 1; i >= 0; i--) {
      Link &link = links[i];
      int parent = link.parent_index;
      link.U = link.IA.mul_inv(link.S);
      link.d = link.S.dot(link.U);
      Scalar tau_val = get_tau_for_link(tau, i);
      // apply linear joint stiffness and damping
      // see Eqns. (2.76), (2.78) in Rigid Body Dynamics Notes
      // by Shinjiro Sueda
      // https://github.com/sueda/redmax/blob/master/notes.pdf
      // TODO consider nonzero resting position of joint for stiffness?
      tau_val -= link.stiffness * get_q_for_link(q, i);
      tau_val -= link.damping * get_qd_for_link(qd, i);

#ifdef NEURAL_SIM
      NEURAL_ASSIGN(tau_val, "tau_" + std::to_string(i));
#endif

      link.u = tau_val - link.S.dot(link.pA);

#ifdef DEBUG
      Algebra::print("m_U", link.U);
      printf("links[%d].d=", i);
      double d1 = Algebra::to_double(link.d);
      printf("%f\n", d1);
      printf("links[%d].u=", i);
      double u = Algebra::to_double(link.u);
      printf("%f\n", u);
#endif

      Scalar invd = link.joint_type == JOINT_FIXED ? Algebra::zero()
                                                   : Algebra::one() / link.d;
#ifdef DEBUG
      printf("invd[%d]=%f\n", i, Algebra::to_double(invd));
#endif
      Inertia tmp = Inertia::vTimesvTranspose(link.U * invd, link.U);

      Inertia Ia = link.IA;
      Ia -= tmp;
      SpatialVector tmp2 = Ia.mul_inv(link.c);
      SpatialVector pa = link.pA + tmp2 + link.U * (link.u * invd);
#ifdef DEBUG
      Algebra::print("Ia-tmp", Ia);
      Algebra::print("tmp2", tmp2);
      Algebra::print("pa", pa);
#endif

      SpatialVector dpA = link.X_parent.apply_transpose(pa);
#ifdef DEBUG
      Algebra::print("dpA", dpA);
#endif
      Inertia dI = Inertia::shift(Ia, link.X_parent);
      if (parent >= 0) {
        links[parent].pA += dpA;
        links[parent].IA += dI;
#ifdef DEBUG
        Algebra::print("pa update", links[parent].pA);
        Algebra::print("mIA", links[parent].I);
#endif
      } else if (is_floating) {
        base_bias_force += dpA;
        base_articulated_inertia += dI;
#ifdef DEBUG
        Algebra::print("base_articulated_inertia", base_articulated_inertia);
        Algebra::print("base_bias_force", base_bias_force);
        Algebra::print("dI", dI);
        Algebra::print("dpA", dpA);
#endif
      }
    }

    if (is_floating) {
#ifdef NEURAL_SIM
      NEURAL_ASSIGN(base_bias_force[0], "base_bias_force_0");
      NEURAL_ASSIGN(base_bias_force[1], "base_bias_force_1");
      NEURAL_ASSIGN(base_bias_force[2], "base_bias_force_2");
      NEURAL_ASSIGN(base_bias_force[3], "base_bias_force_3");
      NEURAL_ASSIGN(base_bias_force[4], "base_bias_force_4");
      NEURAL_ASSIGN(base_bias_force[5], "base_bias_force_5");
#endif

      base_acceleration =
          -base_articulated_inertia.inverse().mul_inv(base_bias_force);

    } else {
      base_acceleration = -spatial_gravity;
    }

    for (int i = 0; i < links.size(); i++) {
      Link &link = links[i];
      int parent = link.parent_index;
      const Transform &X_parent = link.X_parent;
      const SpatialVector &parentAccel =
          (parent >= 0) ? links[parent].a : base_acceleration;
#if DEBUG
      if (parent < 0) {
        printf("final loop for parent %i\n", parent);
        Algebra::print("base_articulated_inertia", base_articulated_inertia);
        Algebra::print("base_bias_force", base_bias_force);
        Algebra::print("parentAccel", parentAccel);
      }
#endif

      SpatialVector xpa = X_parent.apply(parentAccel);
      link.a = xpa + link.c;
#if DEBUG
      Algebra::print("xpa", xpa);
      Algebra::print("a'", link.a);
#endif
      // model.a[i] = X_parent.apply(model.a[parent]) + model.c[i];
      // LOG << "a'[" << i << "] = " << model.a[i].transpose() << std::endl;

      // if (model.mJoints[i].mDoFCount == 1
      {
        Scalar invd = link.joint_type == JOINT_FIXED ? Algebra::zero()
                                                     : Algebra::one() / link.d;
        Scalar t1 = link.U.dot(link.a);
        Scalar t2 = link.u - t1;
        Scalar qdd_val = Algebra::zero();
        if (link.qd_index >= 0) {
          qdd_val = invd * t2;
          qdd[link.qd_index] = qdd_val;
        }
        link.a = link.a + link.S * qdd_val;
        Algebra::print("a", link.a);
      }
    }
    if (is_floating) {
      base_acceleration += spatial_gravity;
      for (int i = 0; i < 6; i++) {
        qdd[i] = base_acceleration[i];
      }
    } else {
      base_acceleration.set_zero();
    }
  }

  void forward_dynamics(const Vector3 &gravity) {
    forward_dynamics(q, qd, tau, gravity, qdd);
  }
};

int main(int argc, char **argv) {
  {
    using Tf = Transform<EnokiAlgebra>;
    Tf tf;
    tf.set_identity();
    std::cout << "tf: " << tf << std::endl;

    EnokiAlgebra::Vector3 vec(1., 2, 3);
    std::cout << "VCM: " << EnokiAlgebra::cross_matrix(vec) << "\n";

    std::cout << "rot-x: " << EnokiAlgebra::rotation_x_matrix(0.3) << std::endl;

    Inertia<EnokiAlgebra> abi;
    std::cout << "ABI: " << abi.inertia << std::endl;

    EnokiAlgebra::Matrix6 mat6(0.);
    EnokiAlgebra::assign_block(mat6, EnokiAlgebra::Matrix3(3.14), 0, 2);
    std::cout << "mat6: " << mat6 << std::endl;
  }

  return 0;

  using namespace enoki;

  using FloatC = CUDAArray<float>;
  using FloatD = DiffArray<FloatC>;

  {
    FloatD a = 1.f;
    set_requires_gradient(a);

    FloatD b = erf(a);
    set_label(a, "a");
    set_label(b, "b");

    // std::cout << graphviz(b) << std::endl;

    backward(b);
    std::cout << gradient(a) << std::endl;
  }

  {
    /* Declare underlying packet type, could just be 'float' for scalar
     * arithmetic
     */
    using FloatP = float;  // Packet<float, 4>;

    /* Define vectorized quaternion type */
    using QuaternionP = Quaternion<FloatP>;

    QuaternionP a = QuaternionP(1.f, 0.f, 0.f, 0.f);
    QuaternionP b = QuaternionP(0.f, 1.f, 0.f, 0.f);

    /* Compute several rotations that interpolate between 'a' and 'b' */
    FloatP t = linspace<FloatP>(0.f, 1.f);
    std::cout << "t:  " << t << std::endl;
    QuaternionP c = slerp(a, b, t);

    std::cout << "Interpolated quaternions:" << std::endl;
    std::cout << c << std::endl << std::endl;

    /* Turn into a 4x4 homogeneous coordinate rotation matrix packet */
    using Matrix4P = Matrix<FloatP, 4>;
    Matrix4P c_rot = quat_to_matrix<Matrix4P>(c);

    std::cout << "Rotation matrices:" << std::endl;
    std::cout << c_rot << std::endl << std::endl;

    /* Round trip: turn the rotation matrices back into rotation quaternions */
    QuaternionP c2 = matrix_to_quat(c_rot);

    if (hsum(abs(c - c2)) < 1e-6f)
      std::cout << "Test passed." << std::endl;
    else
      std::cout << "Test failed." << std::endl;
  }
}