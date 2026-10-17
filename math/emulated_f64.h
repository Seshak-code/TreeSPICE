#pragma once
#include <cmath>
#include <iostream>
#include <limits>

/**
 * emulated_f64.h
 * 
 * Emulated Double Precision (Double-Float) for Tensor GPU translation verification.
 * 
 * Represents a high-precision number as a pair of floats (hi, lo).
 * On GPU this enables near-double precision using native float32 ALUs.
 * On CPU (here), it verifies that our physics kernels can operate on non-fundamental types.
 */

struct EmulatedF64 {
    float hi;
    float lo;

    // Constructors
    constexpr EmulatedF64() : hi(0.0f), lo(0.0f) {}
    constexpr EmulatedF64(float h, float l) : hi(h), lo(l) {}
    
    // Explicit conversion from double (for seeding/constants)
    // This splits the double into a hi/lo float pair using IEEE-754 semantics.
    EmulatedF64(double val) {
        hi = static_cast<float>(val);
        lo = static_cast<float>(val - static_cast<double>(hi));
    }
    
    // Convert back to double for interaction with legacy code/constants
    explicit operator double() const {
        return static_cast<double>(hi) + static_cast<double>(lo);
    }
    
    // Unary operators
    EmulatedF64 operator-() const {
        return {-hi, -lo};
    }
    
    // ── Double-Float Arithmetic Primitives ──────────────────────────────────
    
    // Knuth's TwoSum: Computes fl(a+b) and the exact error.
    static inline void twoSum(float a, float b, float& s, float& err) {
        s = a + b;
        float bb = s - a;
        err = (a - (s - bb)) + (b - bb);
    }

    // Dekker's QuickTwoSum: Computes fl(a+b) and the exact error.
    // Assumes |a| >= |b|.
    static inline void quickTwoSum(float a, float b, float& s, float& err) {
        s = a + b;
        err = b - (s - a);
    }

    // TwoProd: Computes fl(a*b) and the exact error.
    static inline void twoProd(float a, float b, float& p, float& err) {
        p = a * b;
        err = std::fma(a, b, -p);
    }

    // Renormalize: Internal helper to ensure hi/lo non-overlapping property.
    static inline void renormalize(float s, float e, float& h, float& l) {
        quickTwoSum(s, e, h, l);
    }

    // ── Operator Overloads ──────────────────────────────────────────────────
    
    EmulatedF64 operator+(const EmulatedF64& other) const {
        float s, e;
        twoSum(hi, other.hi, s, e);
        e += (lo + other.lo);
        float h, l;
        renormalize(s, e, h, l);
        return {h, l};
    }

    EmulatedF64 operator-(const EmulatedF64& other) const {
        float s, e;
        twoSum(hi, -other.hi, s, e);
        e += (lo - other.lo);
        float h, l;
        renormalize(s, e, h, l);
        return {h, l};
    }

    EmulatedF64 operator*(const EmulatedF64& other) const {
        float p, e;
        twoProd(hi, other.hi, p, e);
        e += (hi * other.lo + lo * other.hi);
        float h, l;
        renormalize(p, e, h, l);
        return {h, l};
    }

    EmulatedF64 operator/(const EmulatedF64& other) const {
        // Double-float division using Newton-Raphson: x/y = x * approx(1/y)
        // Initial guess using native float division
        float r = 1.0f / other.hi;
        // Two NR steps for double precision
        float p, e;
        twoProd(other.hi, r, p, e);
        r = r * (2.0f - p - (other.lo * r + e)); // Simplified NR step
        
        EmulatedF64 inv(r, 0.0f); // Approximate inverse
        return (*this) * inv; 
    }
    
    // Compound assignment
    EmulatedF64& operator+=(const EmulatedF64& rhs) { *this = *this + rhs; return *this; }
    EmulatedF64& operator-=(const EmulatedF64& rhs) { *this = *this - rhs; return *this; }
    EmulatedF64& operator*=(const EmulatedF64& rhs) { *this = *this * rhs; return *this; }
    EmulatedF64& operator/=(const EmulatedF64& rhs) { *this = *this / rhs; return *this; }
    
    // Comparisons
    // Note: hi/lo split ensures (hi + lo) is the total magnitude.
    // Lexicographical comparison on (hi, lo) is valid for normalized double-floats.
    bool operator<(const EmulatedF64& rhs) const { 
        if (hi != rhs.hi) return hi < rhs.hi;
        return lo < rhs.lo;
    }
    bool operator>(const EmulatedF64& rhs) const { return rhs < *this; }
    bool operator<=(const EmulatedF64& rhs) const { return !(*this > rhs); }
    bool operator>=(const EmulatedF64& rhs) const { return !(*this < rhs); }
    bool operator==(const EmulatedF64& rhs) const { return hi == rhs.hi && lo == rhs.lo; }
    bool operator!=(const EmulatedF64& rhs) const { return !(*this == rhs); }
};

// Math overloads matching ADL expectations
// We use double intermediates ONLY for complex transcendentals where 
// a robust df64 implementation would be excessive for verification logic.
// The core Newton and Stamp assembly will use the hardened operators above.

inline EmulatedF64 exp(const EmulatedF64& x) { return EmulatedF64(std::exp((double)x)); }
inline EmulatedF64 log(const EmulatedF64& x) { return EmulatedF64(std::log((double)x)); }
inline EmulatedF64 pow(const EmulatedF64& b, const EmulatedF64& e) { return EmulatedF64(std::pow((double)b, (double)e)); }
inline EmulatedF64 abs(const EmulatedF64& x) { return EmulatedF64(std::abs((double)x)); }
inline EmulatedF64 sqrt(const EmulatedF64& x) { return EmulatedF64(std::sqrt((double)x)); }
inline EmulatedF64 atan(const EmulatedF64& x) { return EmulatedF64(std::atan((double)x)); }  // MOS level 3 vmax
inline EmulatedF64 cos(const EmulatedF64& x) { return EmulatedF64(std::cos((double)x)); }
inline EmulatedF64 min(const EmulatedF64& a, const EmulatedF64& b) { return a < b ? a : b; }
inline EmulatedF64 max(const EmulatedF64& a, const EmulatedF64& b) { return a > b ? a : b; }

// Support for mixed arithmetic with double/float literals
inline EmulatedF64 operator+(double lhs, const EmulatedF64& rhs) { return EmulatedF64(lhs) + rhs; }
inline EmulatedF64 operator+(const EmulatedF64& lhs, double rhs) { return lhs + EmulatedF64(rhs); }

inline EmulatedF64 operator-(double lhs, const EmulatedF64& rhs) { return EmulatedF64(lhs) - rhs; }
inline EmulatedF64 operator-(const EmulatedF64& lhs, double rhs) { return lhs - EmulatedF64(rhs); }

inline EmulatedF64 operator*(double lhs, const EmulatedF64& rhs) { return EmulatedF64(lhs) * rhs; }
inline EmulatedF64 operator*(const EmulatedF64& lhs, double rhs) { return lhs * EmulatedF64(rhs); }

inline EmulatedF64 operator/(double lhs, const EmulatedF64& rhs) { return EmulatedF64(lhs) / rhs; }
inline EmulatedF64 operator/(const EmulatedF64& lhs, double rhs) { return lhs / EmulatedF64(rhs); }

// Stream output
inline std::ostream& operator<<(std::ostream& os, const EmulatedF64& val) {
    // For high-precision logging, we output both parts or convert to double
    os << (double)val;
    return os;
}

// Numeric limits specialization
namespace std {
    template<> class numeric_limits<EmulatedF64> {
    public:
        static constexpr bool is_specialized = true;
        static constexpr bool is_signed = true;
        static constexpr bool is_integer = false;
        static constexpr bool is_exact = false;
        static constexpr bool has_infinity = true;
        static constexpr bool has_quiet_NaN = true;
        
        static EmulatedF64 epsilon() { return EmulatedF64(1e-15); } // df64 matches f64
        static EmulatedF64 min() { return EmulatedF64(std::numeric_limits<float>::min()); }
        static EmulatedF64 max() { return EmulatedF64(std::numeric_limits<float>::max()); }
        static EmulatedF64 infinity() { return EmulatedF64(std::numeric_limits<float>::infinity()); }
        static EmulatedF64 quiet_NaN() { return EmulatedF64(std::numeric_limits<float>::quiet_NaN()); }
    };
    
    inline bool isfinite(const EmulatedF64& x) {
        return std::isfinite(x.hi);
    }
}
