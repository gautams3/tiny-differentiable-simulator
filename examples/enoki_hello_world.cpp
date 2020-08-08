#include <enoki/autodiff.h>
#include <enoki/cuda.h>
#include <enoki/dynamic.h>
#include <enoki/quaternion.h>
#include <enoki/special.h>  // for erf()
#include <fenv.h>

#define USE_MATPLOTLIB 1

#ifdef USE_MATPLOTLIB
#include "third_party/matplotlib-cpp/matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

#define TINY_INLINE ENOKI_INLINE

template <typename Algebra>
struct SpatialVector {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using Vector6 = typename Algebra::Vector6;
  using Matrix6 = typename Algebra::Matrix6;

  Vector3 top{0.};
  Vector3 bottom{0.};

  SpatialVector(const Vector6 &vec)
      : top(vec[0], vec[1], vec[2]), bottom(vec[3], vec[4], vec[5]) {}

  operator Vector6() const {
    return Vector6(top[0], top[1], top[2], bottom[0], bottom[1], bottom[2]);
  }

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

  TINY_INLINE void set_zero() {
    top = 0.;
    bottom = 0.;
  }

  TINY_INLINE SpatialVector &operator+=(const SpatialVector &vec) {
    top += vec.top;
    bottom += vec.bottom;
    return *this;
  }
  TINY_INLINE SpatialVector &operator-=(const SpatialVector &vec) {
    top -= vec.top;
    bottom -= vec.bottom;
    return *this;
  }
  TINY_INLINE SpatialVector &operator*=(const Scalar &s) {
    top *= s;
    bottom *= s;
    return *this;
  }

  TINY_INLINE SpatialVector operator-(const SpatialVector &vec) const {
    return SpatialVector(top - vec.top, bottom - vec.bottom);
  }
  TINY_INLINE SpatialVector operator+(const SpatialVector &vec) const {
    return SpatialVector(top + vec.top, bottom + vec.bottom);
  }
  TINY_INLINE SpatialVector operator-() const {
    return SpatialVector(-top, -bottom);
  }
  TINY_INLINE SpatialVector operator*(const Scalar &s) const {
    return SpatialVector(s * top, s * bottom);
  }

  /**
   * V1 = mv(w1, v1)
   * V2 = mv(w2, v2)
   * V1 x V2 = mv(w1 x w2, w1 x v2 + v1 x w2)
   */
  SpatialVector crossm(const SpatialVector &b) const {
    SpatialVector out;
    out.top = Algebra::cross(top, b.top);
    out.bottom = Algebra::cross(top, b.bottom) + Algebra::cross(bottom, b.top);
    return out;
  }

  /**
   * V = mv(w, v)
   * F = fv(n, f)
   * V x* F = fv(w x n + v x f, w x f)
   */
  SpatialVector crossf(const SpatialVector &b) const {
    SpatialVector out;
    out.top = Algebra::cross(top, b.top) + Algebra::cross(bottom, b.bottom);
    out.bottom = Algebra::cross(top, b.bottom);
    return out;
  }

  friend std::ostream &operator<<(std::ostream &os, const SpatialVector &v) {
    os << "[ " << v.top << "  " << v.bottom << " ]";
    return os;
  }

  ENOKI_STRUCT(SpatialVector, top, bottom)
};
ENOKI_STRUCT_SUPPORT(SpatialVector, top, bottom)

struct EnokiAlgebra {
  using Scalar = double;
  using Vector3 = enoki::Array<Scalar, 3>;
  using Vector6 = enoki::Array<Scalar, 6>;
  using VectorX = std::vector<Scalar>;
  using Matrix3 = enoki::Matrix<Scalar, 3>;
  using Matrix6 = enoki::Matrix<Scalar, 6>;
  using Quaternion = enoki::Quaternion<Scalar>;
  using SpatialVector = ::SpatialVector<EnokiAlgebra>;

  template <typename T>
  ENOKI_INLINE static auto transpose(const T &matrix) {
    return enoki::transpose(matrix);
  }

  template <typename T>
  ENOKI_INLINE static auto inverse(const T &matrix) {
    return enoki::inverse(matrix);
  }

  template <typename T>
  ENOKI_INLINE static auto inverse_transpose(const T &matrix) {
    return enoki::inverse_transpose(matrix);
  }

  template <typename T1, typename T2>
  ENOKI_INLINE static auto cross(const T1 &vector_a, const T2 &vector_b) {
    return enoki::cross(vector_a, vector_b);
  }

  ENOKI_INLINE static Scalar dot(const SpatialVector &a,
                                 const SpatialVector &b) {
    Scalar d = enoki::dot(a.top, b.top);
    d += enoki::dot(a.bottom, b.bottom);
    return d;
  }

  template <typename T1, typename T2>
  ENOKI_INLINE static auto dot(const T1 &vector_a, const T2 &vector_b) {
    return enoki::dot(vector_a, vector_b);
  }

  ENOKI_INLINE static Scalar norm(const SpatialVector &v) {
    Vector6 v6 = v;
    return enoki::norm(v6);
  }

  template <typename T>
  ENOKI_INLINE static Scalar norm(const T &v) {
    return enoki::norm(v);
  }

  template <typename T>
  ENOKI_INLINE static auto normalize(T &v) {
    return enoki::normalize(v);
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
  ENOKI_INLINE static Scalar half() { return 0.5; }

  ENOKI_INLINE static Vector3 zero3() { return Vector3(0); }
  ENOKI_INLINE static Vector3 unit3_x() { return Vector3(1, 0, 0); }
  ENOKI_INLINE static Vector3 unit3_y() { return Vector3(0, 1, 0); }
  ENOKI_INLINE static Vector3 unit3_z() { return Vector3(0, 0, 1); }

  ENOKI_INLINE static Matrix3 zero33() { return Matrix3(0); }

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

  ENOKI_INLINE static Vector3 rotate(const Quaternion &q, const Vector3 &v) {
    return enoki::quat_to_matrix<Matrix3>(q) * v;
  }

  /**
   * Computes the quaternion delta given current rotation q, angular velocity w,
   * time step dt.
   */
  ENOKI_INLINE static Quaternion quat_velocity(const Quaternion &q,
                                               const Vector3 &w,
                                               const Scalar &dt) {
    Quaternion delta(q[3] * w[0] + q[1] * w[2] - q[2] * w[1],
                     q[3] * w[1] + q[2] * w[0] - q[0] * w[2],
                     q[3] * w[2] + q[0] * w[1] - q[1] * w[0],
                     -q[0] * w[0] - q[1] * w[1] - q[2] * w[2]);
    delta *= 0.5 * dt;
    return delta;
  }

  ENOKI_INLINE static const Scalar &quat_x(const Quaternion &q) { return q[0]; }
  ENOKI_INLINE static const Scalar &quat_y(const Quaternion &q) { return q[1]; }
  ENOKI_INLINE static const Scalar &quat_z(const Quaternion &q) { return q[2]; }
  ENOKI_INLINE static const Scalar &quat_w(const Quaternion &q) { return q[3]; }

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

  /**
   * Computes 6x6 matrix by multiplying a and b^T.
   */
  template <std::size_t Size>
  ENOKI_INLINE static enoki::Matrix<Scalar, Size> mul_transpose(
      const enoki::Array<Scalar, Size> &a,
      const enoki::Array<Scalar, Size> &b) {
    enoki::Matrix<Scalar, Size> m(0.0);
    for (std::size_t c = 0; c < Size; ++c) {
      m.col(c) = a * b[c];
    }
    return m;
  }
  ENOKI_INLINE static Matrix6 mul_transpose(const SpatialVector &a,
                                            const SpatialVector &b) {
    Vector6 a6 = a, b6 = b;
    return mul_transpose(a6, b6);
  }

  ENOKI_INLINE friend SpatialVector operator*(const Matrix6 &m,
                                              const SpatialVector &v) {
    Vector6 v6 = v;
    return m * v6;
  }

  EnokiAlgebra() = delete;
};

template <typename Algebra>
struct RigidBodyInertia {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using Matrix3 = typename Algebra::Matrix3;
  using Matrix6 = typename Algebra::Matrix6;
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

  RigidBodyInertia(const RigidBodyInertia<Algebra> &rbi) = default;

  RigidBodyInertia(const Scalar &mass, const Scalar &Ixx, const Scalar &Iyx,
                   const Scalar &Iyy, const Scalar &Izx, const Scalar &Izy,
                   const Scalar &Izz)
      : mass(mass) {
    inertia = {{Ixx, Iyx, Izx}, {Iyx, Iyy, Izy}, {Izx, Izy, Izz}};
  }

  RigidBodyInertia(const Scalar &mass) : mass(mass) {}

  RigidBodyInertia(const Scalar &mass, const Vector3 &com,
                   const Matrix3 &inertia)
      : mass(mass), com(com), inertia(inertia) {}

  RigidBodyInertia(const Scalar &mass, const Vector3 &com, const Scalar &Ixx,
                   const Scalar &Iyx, const Scalar &Iyy, const Scalar &Izx,
                   const Scalar &Izy, const Scalar &Izz)
      : mass(mass), com(com) {
    inertia = {{Ixx, Iyx, Izx}, {Iyx, Iyy, Izy}, {Izx, Izy, Izz}};
  }

  RigidBodyInertia(const Matrix6 &m)
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
  static RigidBodyInertia from_mass_com_i(const Scalar &mass,
                                          const Vector3 &com,
                                          const Matrix3 &inertiaC) {
    const Matrix3 crossCom = Algebra::cross_matrix(com);
    const Matrix3 I = inertiaC + crossCom * Algebra::transpose(crossCom) * mass;
    return RigidBodyInertia(mass, com * mass, I(0, 0), I(1, 0), I(1, 1),
                            I(2, 0), I(2, 1), I(2, 2));
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
  static RigidBodyInertia from_mass_com_xyz(const Scalar &mass,
                                            const Vector3 &com,
                                            const Vector3 &gyrationRadii) {
    return RigidBodyInertia(mass, com * mass, gyrationRadii(0), 0,
                            gyrationRadii(1), 0, 0, gyrationRadii(2));
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

  Matrix6 inverse() const {
    // Inverse of a symmetric block matrix
    // according to (4.1) in
    // http://msvlab.hre.ntou.edu.tw/grades/now/inte/Inverse%20&%20Border/border-LuTT.pdf
    Matrix3 Ainv = Algebra::inverse(inertia);
    Matrix3 B = Algebra::cross_matrix(com);
    Matrix3 C = -B;
    Matrix3 D = Algebra::diagonal3(mass);
    Matrix3 Dinv = Algebra::diagonal3(1.0 / mass);
    Matrix3 DCAB = Algebra::inverse(D - C * Ainv * B);
    Matrix3 AinvBDCAB = Ainv * B * DCAB;

    Matrix6 m;
    Algebra::assign_block(m, Ainv + AinvBDCAB * C * Ainv, 0, 0);
    Algebra::assign_block(m, -AinvBDCAB, 0, 3);
    Algebra::assign_block(m, -DCAB * C * Ainv, 3, 0);
    Algebra::assign_block(m, DCAB, 3, 3);
    return m;
  }

  SpatialVector operator*(const SpatialVector &v) const {
    SpatialVector result;
    result.top = inertia * v.top + Algebra::cross(com, v.bottom);
    result.bottom = mass * v.bottom - Algebra::cross(com, v.top);
    return result;
  }

  RigidBodyInertia operator+(const RigidBodyInertia &rbi) const {
    return Inertia(mass + rbi.mass, com + rbi.com, inertia + rbi.inertia);
  }

  RigidBodyInertia &operator+=(const RigidBodyInertia &rbi) {
    mass += rbi.mass;
    com += rbi.com;
    inertia += rbi.inertia;
    return *this;
  }

  RigidBodyInertia &operator=(const Matrix6 &m) {
    mass = m(3, 3);
    com = Vector3(-m(1, 5), m(0, 5), -m(0, 4));
    Algebra::assign_block(inertia, m, 0, 0, 3, 3);
    return *this;
  }

  ENOKI_STRUCT(RigidBodyInertia, mass, com, inertia)
};
ENOKI_STRUCT_SUPPORT(RigidBodyInertia, mass, com, inertia)

/**
 * The articulated body inertia matrix has the form
 *   [  I   H ]
 *   [ H^T  M ]
 * where M and I are symmetric 3x3 matrices.
 */
template <typename Algebra>
struct ArticulatedBodyInertia {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using Matrix3 = typename Algebra::Matrix3;
  using Matrix6 = typename Algebra::Matrix6;
  typedef ::SpatialVector<Algebra> SpatialVector;
  typedef ::RigidBodyInertia<Algebra> RigidBodyInertia;

  Matrix3 I{Algebra::diagonal3(1.)};
  Matrix3 H{Algebra::zero33()};
  Matrix3 M{Algebra::diagonal3(1.)};

  ArticulatedBodyInertia(const RigidBodyInertia &rbi)
      : I(rbi.inertia),
        H(Algebra::cross_matrix(rbi.com)),
        M(Algebra::diagonal3(rbi.mass)) {}

  ArticulatedBodyInertia &operator=(const RigidBodyInertia &rbi) {
    I = rbi.inertia;
    H = Algebra::cross_matrix(rbi.com);
    M = Algebra::diagonal3(rbi.mass);
    return *this;
  }

  Matrix6 matrix() const {
    Matrix6 m;
    Algebra::assign_block(m, I, 0, 0);
    Algebra::assign_block(m, H, 0, 3);
    Algebra::assign_block(m, Algebra::transpose(H), 3, 0);
    Algebra::assign_block(m, M, 3, 3);
    return m;
  }

  SpatialVector operator*(const SpatialVector &v) const {
    SpatialVector result;
    result.top = I * v.top + H * v.bottom;
    result.bottom = M * v.bottom - Algebra::transpose(H) * v.top;
    return result;
  }

  ArticulatedBodyInertia operator+(const ArticulatedBodyInertia &abi) const {
    return ArticulatedBodyInertia(I + abi.I, H + abi.H, M + abi.M);
  }

  ArticulatedBodyInertia &operator+=(const ArticulatedBodyInertia &abi) {
    I += abi.I;
    H += abi.H;
    M += abi.M;
    return *this;
  }

  ArticulatedBodyInertia &operator+=(const Matrix6 &m) {
    Matrix3 tmp;
    Algebra::assign_block(tmp, m, 0, 0, 3, 3, 0, 0);
    I += tmp;
    Algebra::assign_block(tmp, m, 0, 0, 3, 3, 0, 3);
    H += tmp;
    Algebra::assign_block(tmp, m, 0, 0, 3, 3, 3, 3);
    M += tmp;
    return *this;
  }

  ArticulatedBodyInertia &operator-=(const ArticulatedBodyInertia &abi) {
    I -= abi.I;
    H -= abi.H;
    M -= abi.M;
    return *this;
  }

  ArticulatedBodyInertia &operator-=(const Matrix6 &m) {
    Matrix3 tmp;
    Algebra::assign_block(tmp, m, 0, 0, 3, 3, 0, 0);
    I -= tmp;
    Algebra::assign_block(tmp, m, 0, 0, 3, 3, 0, 3);
    H -= tmp;
    Algebra::assign_block(tmp, m, 0, 0, 3, 3, 3, 3);
    M -= tmp;
    return *this;
  }

  ArticulatedBodyInertia operator+(const Matrix6 &m) const {
    ArticulatedBodyInertia abi(*this);
    abi += m;
    return abi;
  }

  ArticulatedBodyInertia operator-(const Matrix6 &m) const {
    ArticulatedBodyInertia abi(*this);
    abi -= m;
    return abi;
  }

  ArticulatedBodyInertia &operator=(const Matrix6 &m) {
    Algebra::assign_block(I, m, 0, 0, 3, 3, 0, 0);
    Algebra::assign_block(H, m, 0, 0, 3, 3, 0, 3);
    Algebra::assign_block(M, m, 0, 0, 3, 3, 3, 3);
    return *this;
  }

  Matrix6 inverse() const {
    // Inverse of a symmetric block matrix
    // according to (4.1) in
    // http://msvlab.hre.ntou.edu.tw/grades/now/inte/Inverse%20&%20Border/border-LuTT.pdf
    Matrix3 Ainv = Algebra::inverse(I);
    Matrix3 B = H;
    Matrix3 C = -B;
    Matrix3 DCAB = Algebra::inverse(M - C * Ainv * B);
    Matrix3 AinvBDCAB = Ainv * B * DCAB;

    Matrix6 m;
    Algebra::assign_block(m, Ainv + AinvBDCAB * C * Ainv, 0, 0);
    Algebra::assign_block(m, -AinvBDCAB, 0, 3);
    Algebra::assign_block(m, -DCAB * C * Ainv, 3, 0);
    Algebra::assign_block(m, DCAB, 3, 3);
    return m;
  }

  ENOKI_STRUCT(ArticulatedBodyInertia, I, H, M)
};
ENOKI_STRUCT_SUPPORT(ArticulatedBodyInertia, I, H, M)

template <typename Algebra>
struct Transform {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using Matrix3 = typename Algebra::Matrix3;
  using RigidBodyInertia = ::RigidBodyInertia<Algebra>;
  using ArticulatedBodyInertia = ::ArticulatedBodyInertia<Algebra>;
  using SpatialVector = ::SpatialVector<Algebra>;

  Vector3 translation{0.};
  Matrix3 rotation{Matrix3(1.)};

  friend std::ostream &operator<<(std::ostream &os, const Transform &tf) {
    os << "[ translation: " << tf.translation << "  rotation: " << tf.rotation
       << " ]";
    return os;
  }

  TINY_INLINE void set_identity() {
    Algebra::set_zero(translation);
    // set diagonal entries to one, others to zero
    rotation = Matrix3(1.);
  }

  Transform(const Vector3 &translation) : translation(translation) {}
  Transform(const Matrix3 &rotation) : rotation(rotation) {}
  Transform(const Vector3 &translation, const Matrix3 &rotation)
      : translation(translation), rotation(rotation) {}
  Transform(const Scalar &trans_x, const Scalar &trans_y, const Scalar &trans_z)
      : translation(trans_x, trans_y, trans_z) {}

  /**
   * X1*X2 = plx(E1*E2, r2 + E2T*r1)
   */
  Transform operator*(const Transform &t) const {
    Transform tr = *this;
    tr.translation += rotation * t.translation;
    tr.rotation *= t.rotation;
    return tr;
  }

  TINY_INLINE Vector3 apply(const Vector3 &point) const {
    return rotation * point + translation;
  }
  TINY_INLINE Vector3 apply_inverse(const Vector3 &point) const {
    return Algebra::transpose(rotation) * (point - translation);
  }

  Transform inverse() const {
    Transform inv;
    inv.rotation = Algebra::transpose(rotation);
    inv.translation = inv.rotation * -translation;
    return inv;
  }

  /**
   * V = mv(w, v)
   * X*V = mv(E*w, E*(v - r x w))
   */
  inline SpatialVector apply(const SpatialVector &inVec) const {
    SpatialVector outVec;

    Vector3 rxw = Algebra::cross(inVec.top, translation);
    Vector3 v_rxw = inVec.bottom + rxw;

    outVec.top = Algebra::transpose(rotation) * inVec.top;
    outVec.bottom = Algebra::transpose(rotation) * v_rxw;

    return outVec;
  }

  /**
   * V = mv(w, v)
   * inv(X)*V = mv(ET*w, ET*v + r x (ET*w))
   */
  inline SpatialVector apply_inverse(const SpatialVector &inVec) const {
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
  inline SpatialVector apply_transpose(const SpatialVector &inVec) const {
    SpatialVector outVec;
    outVec.bottom = rotation * inVec.bottom;
    outVec.top = rotation * inVec.top;
    outVec.top += Algebra::cross(translation, outVec.bottom);

    return outVec;
  }

  /**
   * F = fv(n, f)
   * X^* F = fv(E(n - rxf), Ef)
   */
  inline SpatialVector apply_inverse_transpose(
      const SpatialVector &inVec) const {
    const Vector3 &n = inVec.top;
    const Vector3 &f = inVec.bottom;
    Matrix3 Et = Algebra::transpose(rotation);
    SpatialVector outVec;
    outVec.top = Et * (n - Algebra::cross(translation, f));
    outVec.bottom = Et * f;
    return outVec;
  }

  /**
   * Computes \f$ X^* I X^{-1} \f$.
   */
  inline RigidBodyInertia apply(const RigidBodyInertia &rbi) const {
    RigidBodyInertia result(rbi.mass);
    const Matrix3 rx = Algebra::cross_matrix(translation);
    const Matrix3 Et = Algebra::transpose(rotation);
    // E(I + rx hx + (h - mr)x rx) E^T
    result.inertia =
        Et *
        (rbi.inertia + rx * Algebra::cross_matrix(rbi.com) +
         Algebra::cross_matrix(rbi.com - rbi.mass * translation) * rx) *
        rotation;
    // E(h - mr)
    result.com = Et * (rbi.com - rbi.mass * translation);
    return result;
  }

  /**
   * Computes \f$ X^T I X \f$.
   */
  inline RigidBodyInertia apply_transpose(const RigidBodyInertia &rbi) const {
    RigidBodyInertia result(rbi.mass);
    // E^T h + mr
    const Matrix3 Et = Algebra::transpose(rotation);
    const Vector3 Et_mr = rotation * rbi.com + rbi.mass * translation;
    const Matrix3 crossTranslation = Algebra::cross_matrix(translation);
    // E^T I E - rx(E^T h)x - (E^T h + mr)x rx
    result.inertia = (rotation * rbi.inertia * Et -
                      crossTranslation * Algebra::cross_matrix(rotation * rbi.com) -
                      Algebra::cross_matrix(Et_mr) * crossTranslation);
    // E^T h + mr
    result.com = Et_mr;
    return result;
  }

  /**
   * Computes \f$ X^* I^A X^{-1} \f$.
   */
  inline ArticulatedBodyInertia apply(const ArticulatedBodyInertia &abi) const {
    ArticulatedBodyInertia result;
    const Matrix3 Et = Algebra::transpose(rotation);
    const Matrix3 rx = Algebra::cross_matrix(translation);
    // H - rx M
    const Matrix3 HrxM = abi.H - rx * abi.M;
    // E (I + rx H^T + (H - rx M) rx) E^T
    result.I =
        Et * (abi.I + rx * Algebra::transpose(abi.H) + HrxM * rx) * rotation;
    // E (H - rx M) E^T
    result.H = Et * HrxM * rotation;
    // E M E^T
    result.M = Et * abi.M * rotation;
    return result;
  }

  /**
   * Computes \f$ X^T I^A X \f$.
   */
  inline ArticulatedBodyInertia apply_transpose(
      const ArticulatedBodyInertia &abi) const {
    ArticulatedBodyInertia result;
    const Matrix3 Et = Algebra::transpose(rotation);
    const Matrix3 rx = Algebra::cross_matrix(translation);
    // M' = E^T M E
    const Matrix3 Mp = rotation * abi.M * Et;
    result.M = Mp;
    // H' = E^T H E
    const Matrix3 Hp = rotation * abi.H * Et;
    // H' + rx M'
    const Matrix3 HrxM = Hp + rx * Mp;
    // E^T I E - rx H'^T - (H' + rx M') rx
    result.I = rotation * abi.I * Et - rx * Algebra::transpose(Hp) - HrxM * rx;
    // H' + rx M'
    result.H = HrxM;
    return result;
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
class Link {
  typedef ::Transform<Algebra> Transform;
  typedef ::SpatialVector<Algebra> SpatialVector;
  typedef ::RigidBodyInertia<Algebra> RigidBodyInertia;
  typedef ::ArticulatedBodyInertia<Algebra> ArticulatedBodyInertia;
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using Matrix3 = typename Algebra::Matrix3;

 public:
  JointType joint_type{JOINT_REVOLUTE_Z};

  Transform X_T;               // parent_link_to_joint
  mutable Transform X_J;       // joint_to_child_link    //depends on q
  mutable Transform X_parent;  // parent_link_to_child_link

  mutable Transform X_world;  // world_to_link
  mutable SpatialVector vJ;   // local joint velocity (relative to parent link)
  mutable SpatialVector v;    // global joint velocity (relative to world)
  mutable SpatialVector a;    // acceleration (relative to world)
  mutable SpatialVector c;    // velocity product acceleration

  RigidBodyInertia rbi;  // local rigid-body spatial inertia (constant)
  mutable ArticulatedBodyInertia
      abi;  // spatial articulated inertia, IC in CRBA

  mutable SpatialVector pA;  // bias forces or zero-acceleration forces
  SpatialVector S;           // motion subspace (spatial joint axis/matrix)

  mutable SpatialVector U;  // temp var in ABA, page 130
  mutable Scalar d;         // temp var in ABA, page 130
  mutable Scalar u;         // temp var in ABA, page 130
  mutable SpatialVector f;  // temp var in RNEA, page 183

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

  Link() = default;
  Link(JointType joint_type, const Transform &parent_link_to_joint,
       const RigidBodyInertia &rbi)
      : X_T(parent_link_to_joint), rbi(rbi) {
    set_joint_type(joint_type);
  }

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
        X_J->rotation = Algebra::rotation_x_matrix(q);
        break;
      case JOINT_REVOLUTE_Y:
        X_J->rotation = Algebra::rotation_y_matrix(q);
        break;
      case JOINT_REVOLUTE_Z:
        X_J->rotation = Algebra::rotation_z_matrix(q);
        break;
      case JOINT_REVOLUTE_AXIS: {
        const Vector3 &axis = S.bottom;
        const auto quat = Algebra::axis_angle_quaternion(axis, q);
        X_J->rotation = Algebra::quat_to_matrix(quat);
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

  inline void jcalc(const Scalar &q) const { jcalc(q, &X_J, &X_parent); }

  inline void jcalc(const Scalar &q, const Scalar &qd) const {
    jcalc(q);
    jcalc(qd, &vJ);
  }
};

template <typename Algebra>
class MultiBody {
  using Scalar = typename Algebra::Scalar;
  using Vector3 = typename Algebra::Vector3;
  using VectorX = typename Algebra::VectorX;
  using Matrix3 = typename Algebra::Matrix3;
  using Matrix6 = typename Algebra::Matrix6;
  using Quaternion = typename Algebra::Quaternion;
  typedef ::Transform<Algebra> Transform;
  typedef ::SpatialVector<Algebra> SpatialVector;
  typedef ::Link<Algebra> Link;
  typedef ::RigidBodyInertia<Algebra> RigidBodyInertia;
  typedef ::ArticulatedBodyInertia<Algebra> ArticulatedBodyInertia;

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
  mutable SpatialVector base_velocity;      // v_0
  mutable SpatialVector base_acceleration;  // a_0
  SpatialVector base_applied_force;         // f_ext_0 in world frame
  mutable SpatialVector base_force;         // f_0 (used by RNEA)
  mutable SpatialVector base_bias_force;    // pA_0
  RigidBodyInertia base_rbi;                // I_0
  mutable ArticulatedBodyInertia base_abi;  // IA_0
  mutable Transform base_X_world;

  std::vector<int> visual_uids1;
  std::vector<int> visual_uids2;
  // offset of geometry (relative to the base frame)
  std::vector<Transform> X_visuals;

  // std::vector<const TinyGeometry<Scalar, Algebra> *>
  //     collision_geometries;
  // offset of collision geometries (relative to this link frame)
  std::vector<Transform> X_collisions;

  VectorX q, qd, qdd, tau;

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
      q.resize(dof());
    }
    for (Scalar &v : q) {
      v = Algebra::zero();
    }
    if (is_floating) {
      q[3] = Algebra::one();  // make sure orientation is valid
    }
    if (static_cast<int>(qd.size()) != dof_qd()) {
      qd.resize(dof_qd());
    }
    for (Scalar &v : qd) {
      v = Algebra::zero();
    }
    if (static_cast<int>(qdd.size()) != dof_qd()) {
      qdd.resize(dof_qd());
    }
    for (Scalar &v : qdd) {
      v = Algebra::zero();
    }
    if (static_cast<int>(tau.size()) != dof_actuated()) {
      tau.resize(dof_actuated());
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
        base_rbi(mb.base_rbi),
        base_X_world(mb.base_X_world),
        // collision_geometries(mb.collision_geometries),
        X_collisions(mb.X_collisions),
        q(mb.q),
        qd(mb.qd),
        qdd(mb.qdd),
        tau(mb.tau) {}

  // virtual ~MultiBody() {
  // if (m_actuator) {
  //   delete actuator;
  // }
  // }

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
      return tf.apply(base_rbi.com);
    } else {
      return tf.apply(links[link].I.com);
    }
  }

  TINY_INLINE Scalar get_q_for_link(const VectorX &q, int link_index) const {
    if (q.empty()) return Algebra::zero();
    const Link &link = links[link_index];
    return link.joint_type == JOINT_FIXED ? Algebra::zero() : q[link.q_index];
  }
  TINY_INLINE Scalar get_q_for_link(int link_index) const {
    get_q_for_link(q, link_index);
  }

  TINY_INLINE Scalar get_qd_for_link(const VectorX &qd, int link_index) const {
    if (qd.empty()) return Algebra::zero();
    const Link &link = links[link_index];
    return link.joint_type == JOINT_FIXED ? Algebra::zero() : qd[link.qd_index];
  }
  TINY_INLINE Scalar get_qd_for_link(int link_index) const {
    return get_qd_for_link(qd, link_index);
  }

  TINY_INLINE Scalar get_qdd_for_link(const VectorX &qdd,
                                      int link_index) const {
    return get_qd_for_link(qdd, link_index);
  }
  TINY_INLINE Scalar get_qdd_for_link(int link_index) const {
    return get_qdd_for_link(qdd, link_index);
  }

  TINY_INLINE Scalar get_tau_for_link(const VectorX &tau,
                                      int link_index) const {
    if (tau.empty()) return Algebra::zero();
    const Link &link = links[link_index];
    int offset = is_floating ? -6 : 0;
    return link.joint_type == JOINT_FIXED ? Algebra::zero()
                                          : tau[link.qd_index + offset];
  }
  TINY_INLINE Scalar get_tau_for_link(int link_index) const {
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
  void forward_kinematics(const VectorX &q, const VectorX &qd = VectorX(),
                          const VectorX &qdd = VectorX()) const {
    assert(static_cast<int>(q.size()) == dof());
    assert(qd.empty() || static_cast<int>(qd.size()) == dof_qd());
    assert(qdd.empty() || static_cast<int>(qdd.size()) == dof_qd());

    if (is_floating) {
      // update base-world transform from q, and update base velocity from qd
      base_X_world.rotation = Algebra::quat_to_matrix(q[0], q[1], q[2], q[3]);
      base_X_world.translation = Vector3(q[4], q[5], q[6]);
      if (!qd.empty()) {
        base_velocity.top = Vector3(qd[0], qd[1], qd[2]);
        base_velocity.bottom = Vector3(qd[3], qd[4], qd[5]);
      } else {
        base_velocity.set_zero();
      }

      SpatialVector I0_mul_v0 = base_rbi * base_velocity;
      base_bias_force = base_velocity.crossf(I0_mul_v0) - base_applied_force;

      base_abi = base_rbi;
    }

    for (int i = 0; i < static_cast<int>(links.size()); i++) {
      const Link &link = links[i];
      int parent = link.parent_index;

      // update joint transforms, joint velocity (if available)
      Scalar q_val = get_q_for_link(q, i);
      Scalar qd_val = get_qd_for_link(qd, i);
      link.jcalc(q_val, qd_val);

      // std::cout << "Link " << i << " transform: " << link.X_parent << std::endl;

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

      link.abi = link.rbi;
      SpatialVector I_mul_v = link.abi * link.v;
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
      Algebra::print("link.abi", link.abi);
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
      link.f = link.abi * link.a + link.pA;
    }
  }

  /**
   * Updates the forward kinematics given the q, qd coordinates stored in this
   * model.
   */
  void forward_kinematics() { forward_kinematics(q, qd); }

  void forward_dynamics(const VectorX &q, const VectorX &qd, const VectorX &tau,
                        const Vector3 &gravity, VectorX &qdd) const {
    assert(static_cast<int>(q.size()) == dof());
    assert(static_cast<int>(qd.size()) == dof_qd());
    assert(static_cast<int>(qdd.size()) == dof_qd());
    assert(static_cast<int>(tau.size()) == dof_actuated());

    SpatialVector spatial_gravity(
        Vector3(Algebra::zero(), Algebra::zero(), Algebra::zero()), gravity);

    // #ifdef NEURAL_SIM
    //     for (int i = 0; i < dof(); ++i) {
    //       NEURAL_ASSIGN(q[i], "q_" + std::to_string(i));
    //     }
    //     for (int i = 0; i < dof_qd(); ++i) {
    //       NEURAL_ASSIGN(qd[i], "qd_" + std::to_string(i));
    //     }
    // #endif

    forward_kinematics(q, qd);

    for (int i = static_cast<int>(links.size()) - 1; i >= 0; i--) {
      const Link &link = links[i];
      int parent = link.parent_index;
      link.U = link.abi * link.S;
      link.d = Algebra::dot(link.S, link.U);
      Scalar tau_val = get_tau_for_link(tau, i);
      // apply linear joint stiffness and damping
      // see Eqns. (2.76), (2.78) in Rigid Body Dynamics Notes
      // by Shinjiro Sueda
      // https://github.com/sueda/redmax/blob/master/notes.pdf
      // TODO consider nonzero resting position of joint for stiffness?
      tau_val -= link.stiffness * get_q_for_link(q, i);
      tau_val -= link.damping * get_qd_for_link(qd, i);

      // #ifdef NEURAL_SIM
      //       NEURAL_ASSIGN(tau_val, "tau_" + std::to_string(i));
      // #endif

      link.u = tau_val - Algebra::dot(link.S, link.pA);

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
      Matrix6 u_dinv_ut = Algebra::mul_transpose(link.U * invd, link.U);

      ArticulatedBodyInertia Ia = link.abi - u_dinv_ut;
      SpatialVector Ia_c = Ia * link.c;
      SpatialVector pa = link.pA + Ia_c + link.U * (link.u * invd);
#ifdef DEBUG
      Algebra::print("Ia", Ia);
      Algebra::print("Ia*c", Ia_c);
      Algebra::print("pa", pa);
#endif

      SpatialVector delta_pA = link.X_parent.apply_transpose(pa);
#ifdef DEBUG
      Algebra::print("delta_pA", delta_pA);
#endif
      ArticulatedBodyInertia delta_I = link.X_parent.apply(Ia);
      if (parent >= 0) {
        links[parent].pA += delta_pA;
        links[parent].abi += delta_I;
#ifdef DEBUG
        Algebra::print("pa update", links[parent].pA);
        Algebra::print("mIA", links[parent].I);
#endif
      } else if (is_floating) {
        base_bias_force += delta_pA;
        base_abi += delta_I;
#ifdef DEBUG
        Algebra::print("base_abi", base_abi);
        Algebra::print("base_bias_force", base_bias_force);
        Algebra::print("delta_I", delta_I);
        Algebra::print("delta_pA", delta_pA);
#endif
      }
    }

    if (is_floating) {
      // #ifdef NEURAL_SIM
      //       NEURAL_ASSIGN(base_bias_force[0], "base_bias_force_0");
      //       NEURAL_ASSIGN(base_bias_force[1], "base_bias_force_1");
      //       NEURAL_ASSIGN(base_bias_force[2], "base_bias_force_2");
      //       NEURAL_ASSIGN(base_bias_force[3], "base_bias_force_3");
      //       NEURAL_ASSIGN(base_bias_force[4], "base_bias_force_4");
      //       NEURAL_ASSIGN(base_bias_force[5], "base_bias_force_5");
      // #endif

      base_acceleration = -base_abi.inverse() * base_bias_force;

    } else {
      base_acceleration = -spatial_gravity;
    }

    for (int i = 0; i < static_cast<int>(links.size()); i++) {
      const Link &link = links[i];
      int parent = link.parent_index;
      const Transform &X_parent = link.X_parent;
      const SpatialVector &parentAccel =
          (parent >= 0) ? links[parent].a : base_acceleration;
#if DEBUG
      if (parent < 0) {
        printf("final loop for parent %i\n", parent);
        Algebra::print("base_abi", base_abi);
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
        Scalar Ut_a = Algebra::dot(link.U, link.a);
        Scalar u_Ut_a = link.u - Ut_a;
        Scalar qdd_val = Algebra::zero();
        if (link.qd_index >= 0) {
          qdd_val = invd * u_Ut_a;
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

  // attaches a new link, setting parent to the last link
  void attach(Link &link, bool is_controllable = true) {
    int parent_index = -1;
    if (!links.empty()) parent_index = static_cast<int>(links.size()) - 1;
    attach(link, parent_index, is_controllable);
  }

  void attach(Link &link, int parent_index, bool is_controllable = true) {
    int sz = static_cast<int>(links.size());
    assert(parent_index < sz);
    link.index = sz;
    link.parent_index = parent_index;
    if (link.joint_type != JOINT_FIXED) {
      assert(Algebra::norm(link.S) > Algebra::zero());
      link.q_index = dof();
      link.qd_index = dof_qd();
      dof_++;
      if (is_controllable) {
        if (control_indices.empty()) {
          control_indices.push_back(0);
        } else {
          control_indices.push_back(control_indices.back() + 1);
        }
      }
    } else {
      link.q_index = -2;
      link.qd_index = -2;
    }
#ifdef DEBUG
    printf(
        "Attached link %i of type %s (parent: %i, index q: %i, index qd: "
        "%i).\n",
        link.rbindex, joint_type_name(link.joint_type).c_str(),
        link.parent_index, link.q_index, link.qd_index);
//    link.S.print("joint.S");
#endif
    links.push_back(link);
  }

  void integrate(VectorX &q, VectorX &qd, VectorX &qdd, const Scalar &dt) {
    assert(static_cast<int>(q.size()) == dof());
    assert(static_cast<int>(qd.size()) == dof_qd());
    assert(static_cast<int>(qdd.size()) == dof_qd());

    int q_offset, qd_offset;
    if (is_floating) {
      base_acceleration.top = Vector3(qdd[0], qdd[1], qdd[2]);
      base_acceleration.bottom = Vector3(qdd[3], qdd[4], qdd[5]);

      base_velocity.top = Vector3(qd[0], qd[1], qd[2]);
      base_velocity.bottom = Vector3(qd[3], qd[4], qd[5]);

      base_velocity += base_acceleration * dt;

      Vector3 linear_velocity = base_velocity.bottom;
      base_X_world.translation += linear_velocity * dt;

      // update base orientation using Quaternion derivative
      Vector3 angular_velocity = base_velocity.top;

      Quaternion base_rot = Algebra::matrix_to_quat(base_X_world.rotation);
      // update 4-dimensional q from 3-dimensional qd for the base rotation
      // angular_velocity = Vector3(qd[0], qd[1], qd[2]);
      base_rot += Algebra::quat_velocity(base_rot, angular_velocity, dt);
      Algebra::normalize(base_rot);
      base_X_world.rotation = Algebra::quat_to_matrix(base_rot);

      q[0] = Algebra::quat_x(base_rot);
      q[1] = Algebra::quat_y(base_rot);
      q[2] = Algebra::quat_z(base_rot);
      q[3] = Algebra::quat_w(base_rot);
      q_offset = 4;
      qd_offset = 3;
    } else {
      q_offset = 0;
      qd_offset = 0;
    }

    for (int i = 0; i < dof_qd() - qd_offset; i++) {
      int qindex = i + q_offset;
      int qdindex = i + qd_offset;
      qd[qdindex] += qdd[qdindex] * dt;
      q[qindex] += qd[qdindex] * dt;
    }
  }

  void integrate(const Scalar &dt) { integrate(q, qd, qdd, dt); }
};

#ifdef USE_MATPLOTLIB
template <typename Algebra>
static void plot_trajectory(
    const std::vector<typename Algebra::VectorX> &states,
    const std::string &title = "Figure") {
  for (int i = 0; i < static_cast<int>(states[0].size()); ++i) {
    std::vector<double> traj(states.size());
    for (int t = 0; t < static_cast<int>(states.size()); ++t) {
      traj[t] = Algebra::to_double(states[t][i]);
    }
    plt::named_plot("state[" + std::to_string(i) + "]", traj);
  }
  plt::legend();
  plt::title(title);
  plt::show();
}
#endif

int main(int argc, char **argv) {
  {
    using Tf = Transform<EnokiAlgebra>;
    using Vector3 = EnokiAlgebra::Vector3;
    using RigidBodyInertia = ::RigidBodyInertia<EnokiAlgebra>;

    // Set NaN trap
    // feenableexcept(FE_INVALID | FE_OVERFLOW);

    Tf tf;
    tf.set_identity();
    std::cout << "tf: " << tf << std::endl;

    EnokiAlgebra::Vector3 vec(1., 2, 3);
    std::cout << "VCM: " << EnokiAlgebra::cross_matrix(vec) << "\n";

    std::cout << "rot-x: " << EnokiAlgebra::rotation_x_matrix(0.3) << std::endl;

    RigidBodyInertia rbi;
    std::cout << "rbi.inertia: " << rbi.inertia << std::endl;

    EnokiAlgebra::Matrix6 mat6(0.);
    EnokiAlgebra::assign_block(mat6, EnokiAlgebra::Matrix3(3.14), 0, 2);
    std::cout << "mat6: " << mat6 << std::endl;

    MultiBody<EnokiAlgebra> mb;

    Link<EnokiAlgebra> link_a(JOINT_REVOLUTE_X, Tf(1., 0., 1.),
                              RigidBodyInertia(1.));
    Link<EnokiAlgebra> link_b(JOINT_REVOLUTE_X, Tf(1., 0., 1.),
                              RigidBodyInertia(1.));
    mb.attach(link_a);
    mb.attach(link_b);
    mb.initialize();

    mb.q = {M_PI_2, 0.0};

    mb.forward_kinematics();
    Vector3 gravity(0., 0., -9.81);
    mb.forward_dynamics(gravity);

    std::vector<typename EnokiAlgebra::VectorX> traj;

    for (int i = 0; i < 1000; ++i) {
      traj.push_back(mb.q);
      mb.integrate(0.01);
      mb.forward_dynamics(gravity);
      mb.print_state();
    }

    plot_trajectory<EnokiAlgebra>(traj);
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