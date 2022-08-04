#include <gtest/gtest.h>
#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"

namespace iree {

using i16 = int16_t;
using u16 = uint16_t;
using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;
using u64 = uint64_t;

constexpr u16 u16_max = std::numeric_limits<u16>::max();

constexpr i32 i32_max = std::numeric_limits<i32>::max();
constexpr i32 i32_min = std::numeric_limits<i32>::min();
constexpr u32 u32_max = std::numeric_limits<u32>::max();

constexpr i64 i64_max = std::numeric_limits<i64>::max();
constexpr i64 i64_min = std::numeric_limits<i64>::min();
constexpr u64 u64_max = std::numeric_limits<u64>::max();

constexpr i32 operator"" _i32(unsigned long long x) { return i32(x); }
constexpr u32 operator"" _u32(unsigned long long x) { return u32(x); }
constexpr i64 operator"" _i64(unsigned long long x) { return i64(x); }
constexpr u64 operator"" _u64(unsigned long long x) { return u64(x); }

template <size_t N, typename T>
using vector = std::array<T, N>;

template <typename T>
constexpr vector<2, T> makeVec(T a, T b) {
  return {a, b};
}

constexpr vector<2, i32> toVec(i64 n) {
  return {i32(u32(u64(n) & u64(u32_max))), i32(u32(u64(n) >> 32))};
}

constexpr i64 toI64(vector<2, i32> n) {
  u64 low = u32(n[0]);
  u64 high = u32(n[1]);
  return i64(low | (high << 32));
}

template <typename T>
constexpr T low(T val) {
  constexpr size_t numBits = sizeof(val) * 8;
  constexpr size_t halfBits = numBits / 2;
  return val & ((T(1) << halfBits) - 1);
}

static_assert(low(u32(u16_max)) == u16_max);

template <typename T>
constexpr T high(T val) {
  constexpr size_t numBits = sizeof(val) * 8;
  constexpr size_t halfBits = numBits / 2;
  return val >> halfBits;
}

static_assert(high(u32(u16_max)) == 0);
static_assert(high(u32_max) == u16_max);

template <typename T>
constexpr T cat(T low, T high) {
  return low | (high << (sizeof(high) * 4));
}

namespace arith {
constexpr vector<2, i32> addi_carry(i32 a, i32 b) {
  u32 low = u32(a) + u32(b);
  i32 carry = (low < u32(a) || low < u32(b)) ? 1_i32 : 0_i32;
  return {i32(low), carry};
}

// Unlike
// [`arith.addi`](https://mlir.llvm.org/docs/Dialects/ArithmeticOps/#arithaddi-mlirarithaddiop),
// this is *not* elementwise.
constexpr vector<2, i32> addi_wide(i32 a, i32 b) {
  vector<2, i32> c = addi_carry(a, b);
  u32 sext = u32(a < 0 ? -1_i32 : 0) + u32(b < 0 ? -1_i32 : 0);
  u32 high = sext + c[1];
  return {c[0], i32(high)};
}

// We could also return `vector<3, i32>`?
constexpr vector<2, i32> addi_wide(vector<2, i32> a, vector<2, i32> b) {
  vector<2, i32> c = addi_carry(a[0], b[0]);
  u32 high = u32(a[1]) + u32(b[1]) + u32(c[1]);
  return {c[0], i32(high)};
}

vector<2, i32> muli(vector<2, i32> x, vector<2, i32> y) {
  vector<4, u32> a = {low(u32(x[0])), high(u32(x[0])), low(u32(x[1])),
                      high(u32(x[1]))};

  vector<4, u32> b = {low(u32(y[0])), high(u32(y[0])), low(u32(y[1])),
                      high(u32(y[1]))};
  constexpr bool debug = false;
  constexpr bool stats = false;

  if (debug) {
    std::cerr << "muli(" << cat(x[0], x[1]) << ", " << cat(y[0], y[1]) << ")\n";
    std::cerr << "a = {" << a[0] << ",\t" << a[1] << ",\t" << a[2] << ",\t"
              << a[3] << "}\n";
    std::cerr << "b = {" << b[0] << ",\t" << b[1] << ",\t" << b[2] << ",\t"
              << b[3] << "}\n";
  }

  unsigned muls = 0;
  unsigned adds = 0;
  struct X {
    unsigned i;
    unsigned j;
    bool low;
  };
  std::map<unsigned, std::vector<X>> deps;
  vector<4, u32> r = {0, 0, 0, 0};

  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 4 - i; ++j) {
      u32 current = r[i + j] + a[i] * b[j];
      ++muls;
      ++adds;
      if (stats) deps[i + j].push_back({i, j, true});
      r[i + j] = low(current);
      if (i + j < 3) {
        r[i + j + 1] += high(current);
        ++adds;
        if (stats) deps[i + j + 1].push_back({i, j, false});
      }
    }

    if (debug) {
      std::cerr << "i = " << i << "\n\tr = {" << r[0] << ",\t" << r[1] << ",\t"
                << r[2] << ",\t" << r[3] << "}\n";
    }
  }

  if (stats) {
    std::cerr << "Muls: " << muls << ", adds: " << adds << "\n";
    std::cerr << "Dependencies:\n";
    for (const auto& [idx, ds] : deps) {
      std::cerr << "\tr[" << idx << "]: " << ds.size() << "\t--\t";
      for (const auto& [f, s, l] : ds)
        std::cerr << (l ? "L" : "H") << "(" << f << ", " << s << "), ";
      std::cerr << "\n";
    }
  }

  vector<2, u32> t = {cat(r[0], r[1]), cat(r[2], r[3])};
  assert(t[0] == u32(x[0]) * u32(y[0]));
  return {i32(t[0]), i32(t[1])};
}

i64 muli(i64 a, i64 b) { return toI64(muli(toVec(a), toVec(b))); }

}  // namespace arith

namespace {

TEST(I64Emulation, Pass) { EXPECT_TRUE(true); }

TEST(I64Emulation, MuliOne) { EXPECT_EQ(arith::muli(13_i64, 37_i64), 481_i64); }

TEST(I64Emulation, Muli) {
  EXPECT_EQ(arith::muli(0_i64, 0_i64), 0_i64);
  EXPECT_EQ(arith::muli(1_i64, 0_i64), 0_i64);
  EXPECT_EQ(arith::muli(1_i64, 1_i64), 1_i64);
  EXPECT_EQ(arith::muli(makeVec(1_i32, 0_i32), makeVec(i32(u16_max), 0)),
            makeVec(i32(u16_max), 0_i32));
  EXPECT_EQ(arith::muli(-3_i64, -57_i64), -3_i64 * -57_i64);
  EXPECT_EQ(arith::muli(4_i64, i64(u16_max)), 4_i64 * i64(u16_max));
  EXPECT_EQ(arith::muli(-4_i64, i64(u16_max)), -4_i64 * i64(u16_max));
  EXPECT_EQ(arith::muli(i64(u16_max), i64(u16_max)),
            i64(u16_max) * i64(u16_max));
}

TEST(I64Emulation, MuliRand) {
  std::default_random_engine engine(0);
  std::uniform_int_distribution<u64> dist;
  for (size_t i = 0; i < 1000000; ++i) {
    u64 a = dist(engine);
    u64 b = dist(engine);
    ASSERT_EQ(u64(arith::muli(a, b)), a * b);
    u64 m = a * b;
    // Karatsuba.
    u64 z0 = low(a) * low(b);
    u64 z2 = high(a) * high(b);
    u64 z1 = u64(low(a) + high(a)) * u64(low(b) + high(b)) - z2 - z0;
    u64 res = z0 + (z1 << 32);
    ASSERT_EQ(res, m);
  }
}

TEST(I64Emulation, ToVec) {
  EXPECT_EQ(toVec(0_i64), makeVec(0_i32, 0_i32));
  EXPECT_EQ(toVec(i32_max), makeVec(i32_max, 0_i32));
  EXPECT_EQ(toVec(i64(u64(u32_max) << 32)), makeVec(0_i32, i32(u32_max)));
  EXPECT_EQ(toVec(i64(u64_max)), makeVec(i32(u32_max), i32(u32_max)));
}

TEST(I64Emulation, ToI64) {
  for (i64 n :
       {0_i64, 2_i64, -1_i64, i64(i32_max), 1_i64 + i32_max, i64(i32_min),
        i64(i32_min) - 1, i64_min, i64_min + 1, i64_max, i64_max - 1}) {
    EXPECT_EQ(n, toI64(toVec(n)));
  }
}

TEST(I64Emulation, AddICarry) {
  EXPECT_EQ(arith::addi_wide(0_i32, 0_i32), makeVec(0_i32, 0_i32));
  EXPECT_EQ(arith::addi_wide(1_i32, 2_i32), makeVec(3_i32, 0_i32));
  EXPECT_EQ(arith::addi_wide(1_i32, -1_i32), makeVec(0_i32, 0_i32));
  EXPECT_EQ(arith::addi_wide(i32_max, 0_i32), makeVec(i32_max, 0_i32));
  EXPECT_EQ(arith::addi_wide(i32_max, 1_i32), makeVec(i32_min, 0_i32));
  EXPECT_EQ(arith::addi_wide(i32_max, i32_min), makeVec(-1_i32, -1_i32));
  EXPECT_EQ(arith::addi_wide(-1_i32, -1_i32), makeVec(-2_i32, -1_i32));
}

TEST(I64Emulation, AddICarryRoundTrip) {
  for (i32 a : {i32_min, -13_i32, -1_i32, 0_i32, 1_i32, 13_i32, i32_max})
    for (i32 b : {i32_min, -13_i32, -1_i32, 0_i32, 1_i32, 13_i32, i32_max})
      EXPECT_EQ(arith::addi_wide(a, b), toVec(i64(a) + i64(b)))
          << "a: " << a << ", b: " << b;
}

TEST(I64Emulation, AddICarryVecRoundTrip) {
  std::initializer_list<i64> nums = {
      i64(i32_min), -13_i64, -1_i64, 0_i64, 1_i64, i64(i32_max), i64(i32_max)};
  for (i64 a : nums)
    for (i32 b : nums)
      EXPECT_EQ(arith::addi_wide(toVec(a), toVec(b)), toVec(a + b))
          << "a: " << a << ", b: " << b;
}

}  // namespace
}  // namespace iree
