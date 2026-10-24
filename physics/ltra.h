#pragma once
// ngspice 47's lossy transmission line LTRA (devices/ltra, J. S. Roychowdhury 1990): the RLC, RC, LC and RG cases,
// the convolution coefficients over the accepted time points (piecewise-linear history), the inputs that go into
// the branch rows' right-hand sides, and the per-step limits. Ported as written, its numerical choices included
// (the Bessel approximations, the coefficient cut-off, the maxSafeStep search).
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

namespace ltra {

enum class Case { LC, RLC, RC, RG };
enum class Interp { LIN, QUAD, MIXED };

struct Model {
    Case kase = Case::LC;
    Interp interp = Interp::QUAD;
    double R = 0.0, L = 0.0, G = 0.0, C = 0.0, len = 0.0;
    double imped = 0.0, admit = 0.0, td = 0.0, alpha = 0.0, beta = 0.0, attenuation = 0.0;
    double intH1dash = 0.0, intH2 = 0.0, intH3dash = 0.0;
    double cByR = 0.0, rclsqr = 0.0;
    double coshlrootGR = 0.0, rRsLrGRorG = 0.0, rGsLrGRorR = 0.0;  // RG: its two-port (ltraload MODEDC)
    double maxSafeStep = 1e300;
    double chopReltol = 0.0;            // the coefficients' cut-off (ngspice LTRAchopReltol, 0 by default)
    double reltol = 1.0, abstol = 1.0;  // the breakpoints' rate-of-change tolerances (ngspice "rel", "abs")
    bool stepLimit = true, truncDontCut = false;
};

// The accepted history (t[0] = 0) and the line's initial state, plus a cache of the inputs at one trial time
struct History {
    std::vector<double> t, v1, v2, i1, i2;
    double iv1 = 0.0, iv2 = 0.0, ii1 = 0.0, ii2 = 0.0;
    std::vector<double> breaks;  // breakpoints the accepted points set (sorted)
    mutable double cacheT = -1.0;  // the inputs at one trial time (ltraload computes them once per time point)
    mutable size_t cacheN = 0;
    mutable double h1f = 0.0, h2f = 0.0, h3f = 0.0, in1 = 0.0, in2 = 0.0;
};

namespace detail {
inline double bessI0(double x) {
    double ax, ans, y;
    if ((ax = std::fabs(x)) < 3.75) {
        y = x / 3.75; y *= y;
        ans = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))));
    } else {
        y = 3.75 / ax;
        ans = (std::exp(ax) / std::sqrt(ax)) * (0.39894228 + y * (0.1328592e-1 + y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2
              + y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1 + y * 0.392377e-2))))))));
    }
    return ans;
}
inline double bessI1(double x) {
    double ax, ans, y;
    if ((ax = std::fabs(x)) < 3.75) {
        y = x / 3.75; y *= y;
        ans = ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934 + y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3))))));
    } else {
        y = 3.75 / ax;
        ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1 - y * 0.420059e-2));
        ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2 + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))));
        ans *= (std::exp(ax) / std::sqrt(ax));
    }
    return x < 0.0 ? -ans : ans;
}
inline double bessI1xOverX(double x) {
    double ax, ans, y;
    if ((ax = std::fabs(x)) < 3.75) {
        y = x / 3.75; y *= y;
        ans = 0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934 + y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3)))));
    } else {
        y = 3.75 / ax;
        ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1 - y * 0.420059e-2));
        ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2 + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))));
        ans *= (std::exp(ax) / (ax * std::sqrt(ax)));
    }
    return ans;
}
inline double intlinfunc(double lolimit, double hilimit, double lovalue, double hivalue, double t1, double t2) {
    const double width = t2 - t1;
    if (width == 0.0) return 0.0;
    const double m = (hivalue - lovalue) / width;
    return (hilimit - lolimit) * lovalue + 0.5 * m * ((hilimit - t1) * (hilimit - t1) - (lolimit - t1) * (lolimit - t1));
}
inline double twiceintlinfunc(double lolimit, double hilimit, double otherlolimit, double lovalue, double hivalue, double t1, double t2) {
    const double width = t2 - t1;
    if (width == 0.0) return 0.0;
    const double m = (hivalue - lovalue) / width;
    const double temp1 = hilimit - t1, temp2 = lolimit - t1, temp3 = otherlolimit - t1;
    double dummy = lovalue * ((hilimit - otherlolimit) * (hilimit - otherlolimit) - (lolimit - otherlolimit) * (lolimit - otherlolimit));
    dummy += m * ((temp1 * temp1 * temp1 - temp2 * temp2 * temp2) / 3.0 - temp3 * temp3 * (hilimit - lolimit));
    return dummy * 0.5;
}
inline double rlcH2(double time, double T, double alpha, double beta) {
    if (alpha == 0.0 || time < T) return 0.0;
    const double besselarg = time != T ? alpha * std::sqrt(time * time - T * T) : 0.0;
    return alpha * alpha * T * std::exp(-beta * time) * bessI1xOverX(besselarg);
}
inline double rlcH3dash(double time, double T, double alpha, double beta) {
    if (alpha == 0.0 || time < T) return 0.0;
    const double besselarg = time != T ? alpha * std::sqrt(time * time - T * T) : 0.0;
    return (alpha * time * bessI1xOverX(besselarg) - bessI0(besselarg)) * alpha * std::exp(-beta * time);
}
inline bool straightLine(double x1, double y1, double x2, double y2, double x3, double y3, double reltol, double abstol) {
    const double q1 = (std::fabs(y2) + std::fabs(y1)) * 0.5 * std::fabs(x2 - x1), q2 = (std::fabs(y3) + std::fabs(y2)) * 0.5 * std::fabs(x3 - x2);
    const double q3 = (std::fabs(y3) + std::fabs(y1)) * 0.5 * std::fabs(x3 - x1);
    return (q1 + q2) * reltol + abstol > std::fabs(q3 - q1 - q2);
}
// ngspice LTRAquadInterp / LTRAlinInterp: the weights of t's neighbours
inline bool quadInterp(double t, double t1, double t2, double t3, double &c1, double &c2, double &c3) {
    if (t == t1) { c1 = 1; c2 = 0; c3 = 0; return true; }
    if (t == t2) { c1 = 0; c2 = 1; c3 = 0; return true; }
    if (t == t3) { c1 = 0; c2 = 0; c3 = 1; return true; }
    if ((t2 - t1) == 0 || (t3 - t2) == 0 || (t1 - t3) == 0) return false;
    double f1 = (t - t2) * (t - t3), f2 = (t - t1) * (t - t3), f3 = (t - t1) * (t - t2);
    f1 /= (t1 - t2); f2 /= (t2 - t1);
    f2 /= (t2 - t3); f3 /= (t2 - t3);
    f1 /= (t1 - t3); f3 /= (t1 - t3);
    c1 = f1; c2 = f2; c3 = f3;
    return true;
}
inline bool linInterp(double t, double t1, double t2, double &c1, double &c2) {
    if (t1 == t2) return false;
    if (t == t1) { c1 = 1; c2 = 0; return true; }
    if (t == t2) { c1 = 0; c2 = 1; return true; }
    const double temp = (t - t1) / (t2 - t1);
    c2 = temp; c1 = 1 - temp;
    return true;
}
}  // namespace detail

// ngspice ltraset / ltratemp: the case from which of R, L, G, C are nonzero, and its constants. ckt: .options
// reltol / abstol (the straight-line check of maxSafeStep). Throws ngspice's errors.
// stReltol / stAbstol: the straight-line tolerances (compactrel / compactabs, else .options reltol / abstol)
inline void setup(Model &m, const std::string &name, double stReltol, double stAbstol) {
    using namespace detail;
    const bool r = m.R != 0.0, l = m.L != 0.0, g = m.G != 0.0, c = m.C != 0.0;
    if (!(m.len > 0.0)) throw std::runtime_error("model " + name + ": lossy line length LEN must be given and > 0");
    if ((int)r + (int)l + (int)g + (int)c <= 1) throw std::runtime_error("model " + name + ": at least two of R, L, G, C must be given and nonzero");
    if (g && (c || l)) throw std::runtime_error("model " + name + ": a nonzero G (except an RG line) is not supported (as ngspice)");
    if (r && l && !c) throw std::runtime_error("model " + name + ": an RL line is not supported (as ngspice)");
    if (!r && c && l) m.kase = Case::LC;
    else if (r && c && l) m.kase = Case::RLC;
    else if (r && c && !l) m.kase = Case::RC;
    else if (r && g) m.kase = Case::RG;
    else throw std::runtime_error("model " + name + ": this R, L, G, C combination is not a lossy line ngspice supports");
    switch (m.kase) {
    case Case::LC:
        m.imped = std::sqrt(m.L / m.C); m.admit = 1.0 / m.imped; m.td = std::sqrt(m.L * m.C) * m.len; m.attenuation = 1.0;
        break;
    case Case::RLC: {
        m.imped = std::sqrt(m.L / m.C); m.admit = 1.0 / m.imped; m.td = std::sqrt(m.L * m.C) * m.len;
        m.alpha = 0.5 * (m.R / m.L); m.beta = m.alpha;
        m.attenuation = std::exp(-m.beta * m.td);
        if (m.alpha > 0.0) { m.intH1dash = -1.0; m.intH2 = 1.0 - m.attenuation; m.intH3dash = -m.attenuation; }
        if (!m.truncDontCut) {  // ltratemp: as written (its second check repeats the first)
            double xbig = m.td + 9 * m.td, xsmall = m.td, xmid = 0.5 * (xbig + xsmall);
            const double y1small = rlcH2(xsmall, m.td, m.alpha, m.beta);
            for (int iters = 1;; ++iters) {
                const double y1big = rlcH2(xbig, m.td, m.alpha, m.beta), y1mid = rlcH2(xmid, m.td, m.alpha, m.beta);
                const int done = straightLine(xbig, y1big, xmid, y1mid, xsmall, y1small, stReltol, stAbstol) * 2;
                if (done == 2 || iters > 50) break;
                xbig = xmid; xmid = 0.5 * (xbig + xsmall);
            }
            m.maxSafeStep = xbig - m.td;
        }
        break;
    }
    case Case::RC:
        m.cByR = m.C / m.R; m.rclsqr = m.R * m.C * m.len * m.len;
        m.intH1dash = 0.0; m.intH2 = 1.0; m.intH3dash = 0.0;
        break;
    case Case::RG: {
        const double d1 = m.len * std::sqrt(m.R * m.G), e2 = std::exp(-d1), e1 = std::exp(d1);
        m.coshlrootGR = 0.5 * (e1 + e2);
        m.rRsLrGRorG = m.G <= 1e-10 ? m.len * m.R : 0.5 * (e1 - e2) * std::sqrt(m.R / m.G);
        m.rGsLrGRorR = m.R <= 1e-10 ? m.len * m.G : 0.5 * (e1 - e2) * std::sqrt(m.G / m.R);
        break;
    }
    }
}

// The inputs at trial time t over h's accepted points: the first coefficients (the current point's, in the
// matrix) and in1 / in2 for the branch rows (ngspice ltraload's INITPRED part, cached per t and history length)
inline void inputs(const Model &m, const History &h, double t) {
    using namespace detail;
    const size_t n = h.t.size();
    if (h.cacheT == t && h.cacheN == n) return;
    h.cacheT = t; h.cacheN = n;
    const int ti = static_cast<int>(n) - 1;  // the last accepted point
    const auto &tl = h.t;
    h.in1 = h.in2 = h.h1f = h.h2f = h.h3f = 0.0;
    std::vector<double> c1(n, 0.0), c2(n, 0.0), c3(n, 0.0);
    int aux = 0;
    if (m.kase == Case::RC) {  // LTRArcCoeffsSetup
        double delta1 = t - tl[ti], lo1 = 0.0, hi1 = delta1;
        double h1lo1 = 0.0, h1hi1 = std::sqrt(4 * m.cByR * hi1 / M_PI), h1d1 = h1hi1 / delta1;
        h.h1f = h1d1;
        const double h1rel = std::fabs(h1d1 * m.chopReltol);
        double temp = m.rclsqr / (4 * hi1), temp2 = temp >= 100.0 ? 0.0 : std::erfc(std::sqrt(temp)), temp3 = std::exp(-temp);
        const double temp4 = std::sqrt(m.rclsqr), temp5 = std::sqrt(m.cByR);
        double h2lo1 = 0.0, h2hi1 = hi1 != 0.0 ? (hi1 + m.rclsqr * 0.5) * temp2 - std::sqrt(hi1 * m.rclsqr / M_PI) * temp3 : 0.0;
        double h2d1 = h2hi1 / delta1;
        h.h2f = h2d1;
        const double h2rel = std::fabs(h2d1 * m.chopReltol);
        double h3lo1 = 0.0, h3hi1 = hi1 != 0.0 ? temp5 * (2 * std::sqrt(hi1 / M_PI) * temp3 - temp4 * temp2) : 0.0;
        double h3d1 = h3hi1 / delta1;
        h.h3f = h3d1;
        const double h3rel = std::fabs(h3d1 * m.chopReltol);
        bool doh1 = true, doh2 = true, doh3 = true;
        (void)lo1; (void)h1lo1; (void)h2lo1; (void)h3lo1;
        for (int i = ti; i > 0; --i) {
            const double hi2 = hi1;
            delta1 = tl[i] - tl[i - 1];
            lo1 = hi2;
            hi1 = t - tl[i - 1];
            if (doh1) {
                const double h1hi2 = h1hi1, h1d2 = h1d1;
                h1lo1 = h1hi2; h1hi1 = std::sqrt(4 * m.cByR * hi1 / M_PI);
                h1d1 = (h1hi1 - h1lo1) / delta1;
                c1[i] = h1d1 - h1d2;
                if (std::fabs(c1[i]) < h1rel) doh1 = false;
            }
            if (doh2 || doh3) {
                temp = m.rclsqr / (4 * hi1); temp2 = temp >= 100.0 ? 0.0 : std::erfc(std::sqrt(temp)); temp3 = std::exp(-temp);
            }
            if (doh2) {
                const double h2hi2 = h2hi1, h2d2 = h2d1;
                h2lo1 = h2hi2;
                h2hi1 = hi1 != 0.0 ? (hi1 + m.rclsqr * 0.5) * temp2 - std::sqrt(hi1 * m.rclsqr / M_PI) * temp3 : 0.0;
                h2d1 = (h2hi1 - h2lo1) / delta1;
                c2[i] = h2d1 - h2d2;
                if (std::fabs(c2[i]) < h2rel) doh2 = false;
            }
            if (doh3) {
                const double h3hi2 = h3hi1, h3d2 = h3d1;
                h3lo1 = h3hi2;
                h3hi1 = hi1 != 0.0 ? temp5 * (2 * std::sqrt(hi1 / M_PI) * temp3 - temp4 * temp2) : 0.0;
                h3d1 = (h3hi1 - h3lo1) / delta1;
                c3[i] = h3d1 - h3d2;
                if (std::fabs(c3[i]) < h3rel) doh3 = false;
            }
        }
    } else if (m.kase == Case::RLC) {  // LTRArlcCoeffsSetup
        const double T = m.td, alpha = m.alpha, beta = m.beta;
        if (T == 0.0) aux = ti;
        else if (t - T <= 0.0) aux = 0;
        else {
            bool exact = false;
            int i;
            for (i = ti; i >= 0; --i) {
                if (t - tl[i] == T) { exact = true; break; }
                if (t - tl[i] > T) break;
            }
            if (i < 0 || (i == 0 && exact)) throw std::runtime_error("LTRA: cannot place the delayed point (ngspice: some mistake)");
            aux = exact ? i - 1 : i;
        }
        double lo1, hi1, delta1, besselarg = 0.0, exparg, expterm, alphasqT = 0.0, expbetaT = 0.0;
        double h2lo1 = 0.0, h2hi1 = 0.0, h2d1 = 0.0, h3lo1 = 0.0, h3hi1 = 0.0, h3d1 = 0.0, h2rel = 0.0, h3rel = 0.0;
        if (aux != 0) {
            lo1 = T; hi1 = t - tl[aux]; delta1 = hi1 - lo1;
            h2lo1 = rlcH2(T, T, alpha, beta);
            besselarg = hi1 > T ? alpha * std::sqrt(hi1 * hi1 - T * T) : 0.0;
            expterm = std::exp(-beta * hi1);
            alphasqT = alpha * alpha * T;
            h2hi1 = (alpha == 0.0 || hi1 < T) ? 0.0 : alphasqT * expterm * bessI1xOverX(besselarg);
            h2d1 = twiceintlinfunc(lo1, hi1, lo1, h2lo1, h2hi1, lo1, hi1) / delta1;
            h.h2f = h2d1;
            h2rel = std::fabs(m.chopReltol * h2d1);
            h3lo1 = 0.0;
            expbetaT = std::exp(-beta * T);
            h3hi1 = (hi1 <= T || beta == 0.0) ? 0.0 : expterm * bessI0(besselarg) - expbetaT;
            h3d1 = intlinfunc(lo1, hi1, h3lo1, h3hi1, lo1, hi1) / delta1;
            h.h3f = h3d1;
            h3rel = std::fabs(h3d1 * m.chopReltol);
        }
        lo1 = 0.0; hi1 = t - tl[ti]; delta1 = hi1 - lo1;
        exparg = -beta * hi1; expterm = std::exp(exparg);
        double h1lo1 = 0.0;
        double h1hi1 = beta == 0.0 ? hi1 : (hi1 == 0.0 ? 0.0 : (bessI1(-exparg) + bessI0(-exparg)) * hi1 * expterm - hi1);
        double h1d1 = h1hi1 / delta1;
        h.h1f = h1d1;
        const double h1rel = std::fabs(h1d1 * m.chopReltol);
        bool doh1 = true, doh2 = true, doh3 = true;
        double lo2 = 0.0, hi2 = 0.0;
        for (int i = ti; i > 0; --i) {
            if (doh1 || doh2 || doh3) {
                lo2 = lo1; hi2 = hi1;
                lo1 = hi2; hi1 = t - tl[i - 1];
                delta1 = tl[i] - tl[i - 1];
                exparg = -beta * hi1; expterm = std::exp(exparg);
            }
            if (doh1) {
                const double h1hi2 = h1hi1, h1d2 = h1d1;
                h1lo1 = h1hi2;
                h1hi1 = beta == 0.0 ? hi1 : (hi1 == 0.0 ? 0.0 : (bessI1(-exparg) + bessI0(-exparg)) * hi1 * expterm - hi1);
                h1d1 = (h1hi1 - h1lo1) / delta1;
                c1[i] = h1d1 - h1d2;
                if (std::fabs(c1[i]) <= h1rel) doh1 = false;
            }
            if (i <= aux) {
                if (doh2 || doh3) besselarg = hi1 > T ? alpha * std::sqrt(hi1 * hi1 - T * T) : 0.0;
                if (doh2) {
                    const double h2lo2 = h2lo1, h2hi2 = h2hi1, h2d2 = h2d1;
                    h2lo1 = h2hi2;
                    h2hi1 = (alpha == 0.0 || hi1 < T) ? 0.0 : alphasqT * expterm * bessI1xOverX(besselarg);
                    h2d1 = twiceintlinfunc(lo1, hi1, lo1, h2lo1, h2hi1, lo1, hi1) / delta1;
                    c2[i] = h2d1 - h2d2 + intlinfunc(lo2, hi2, h2lo2, h2hi2, lo2, hi2);
                    if (std::fabs(c2[i]) <= h2rel) doh2 = false;
                }
                if (doh3) {
                    const double h3hi2 = h3hi1, h3d2 = h3d1;
                    h3lo1 = h3hi2;
                    h3hi1 = (hi1 <= T || beta == 0.0) ? 0.0 : expterm * bessI0(besselarg) - expbetaT;
                    h3d1 = intlinfunc(lo1, hi1, h3lo1, h3hi1, lo1, hi1) / delta1;
                    c3[i] = h3d1 - h3d2;
                    if (std::fabs(c3[i]) <= h3rel) doh3 = false;
                }
            }
        }
    }

    // the delayed values (LC, RLC): interpolated at t - TD between the accepted points around it
    const bool tdover = (m.kase == Case::LC || m.kase == Case::RLC) && t > m.td;
    double v1d = 0.0, v2d = 0.0, i1d = 0.0, i2d = 0.0;
    if (tdover) {
        int i;
        for (i = ti; i >= 0; --i) if (tl[i] < t - m.td) break;
        if (i == ti) --i;
        if (i == -1) throw std::runtime_error("LTRA: cannot find the delayed time point (ngspice: mistake)");
        const double t2 = tl[i], t3 = tl[i + 1];
        const bool quad = m.interp == Interp::QUAD || m.interp == Interp::MIXED;
        double qf1 = 0, qf2 = 0, qf3 = 0, lf2 = 0, lf3 = 0;
        if (i != 0 && quad) quadInterp(t - m.td, tl[i - 1], t2, t3, qf1, qf2, qf3);
        if (i == 0 || m.interp == Interp::MIXED || m.interp == Interp::LIN) linInterp(t - m.td, t2, t3, lf2, lf3);
        auto delayed = [&](const std::vector<double> &x) {
            double d = 0.0, mx = 0.0, mn = 0.0;
            if (i != 0 && quad) {
                d = x[i - 1] * qf1 + x[i] * qf2 + x[i + 1] * qf3;
                mx = std::max({x[i - 1], x[i], x[i + 1]}); mn = std::min({x[i - 1], x[i], x[i + 1]});
            }
            // linear when asked, at the first interval, or for MIXED when the parabola overshoots; QUAD keeps it
            if (m.interp == Interp::LIN || i == 0 || (i != 0 && quad && (d > mx || d < mn)))
                if (!(i != 0 && m.interp == Interp::QUAD)) d = x[i] * lf2 + x[i + 1] * lf3;
            return d;
        };
        v1d = delayed(h.v1); i1d = delayed(h.i1); v2d = delayed(h.v2); i2d = delayed(h.i2);
    }

    double &in1 = h.in1, &in2 = h.in2;  // mutable
    const double iv1 = h.iv1, iv2 = h.iv2, ii1 = h.ii1, ii2 = h.ii2;
    if (m.kase == Case::RLC) {
        double d1 = 0.0, d2 = 0.0;
        for (int i = ti; i > 0; --i)
            if (c1[i] != 0.0) { d1 += c1[i] * (h.v1[i] - iv1); d2 += c1[i] * (h.v2[i] - iv2); }
        d1 += iv1 * m.intH1dash - iv1 * h.h1f; d2 += iv2 * m.intH1dash - iv2 * h.h1f;
        in1 -= d1 * m.admit; in2 -= d2 * m.admit;
        d1 = d2 = 0.0;
        if (tdover) {
            d1 = (i2d - ii2) * h.h2f; d2 = (i1d - ii1) * h.h2f;
            for (int i = aux; i > 0; --i)
                if (c2[i] != 0.0) { d1 += c2[i] * (h.i2[i] - ii2); d2 += c2[i] * (h.i1[i] - ii1); }
        }
        d1 += ii2 * m.intH2; d2 += ii1 * m.intH2;
        in1 += d1; in2 += d2;
        d1 = d2 = 0.0;
        if (tdover) {
            d1 = (v2d - iv2) * h.h3f; d2 = (v1d - iv1) * h.h3f;
            for (int i = aux; i > 0; --i)
                if (c3[i] != 0.0) { d1 += c3[i] * (h.v2[i] - iv2); d2 += c3[i] * (h.v1[i] - iv1); }
        }
        d1 += iv2 * m.intH3dash; d2 += iv1 * m.intH3dash;
        in1 += m.admit * d1; in2 += m.admit * d2;
    }
    if (m.kase == Case::RLC || m.kase == Case::LC) {  // the lossless-like parts
        if (!tdover) { in1 += m.attenuation * (iv2 * m.admit + ii2); in2 += m.attenuation * (iv1 * m.admit + ii1); }
        else { in1 += m.attenuation * (v2d * m.admit + i2d); in2 += m.attenuation * (v1d * m.admit + i1d); }
    }
    if (m.kase == Case::RC) {
        double d1 = 0.0, d2 = 0.0;
        for (int i = ti; i > 0; --i)
            if (c1[i] != 0.0) { d1 += c1[i] * (h.v1[i] - iv1); d2 += c1[i] * (h.v2[i] - iv2); }
        d1 += iv1 * m.intH1dash - iv1 * h.h1f; d2 += iv2 * m.intH1dash - iv2 * h.h1f;
        in1 -= d1; in2 -= d2;
        d1 = d2 = 0.0;
        for (int i = ti; i > 0; --i)
            if (c2[i] != 0.0) { d1 += c2[i] * (h.i2[i] - ii2); d2 += c2[i] * (h.i1[i] - ii1); }
        d1 += ii2 * m.intH2 - ii2 * h.h2f; d2 += ii1 * m.intH2 - ii1 * h.h2f;
        in1 += d1; in2 += d2;
        d1 = d2 = 0.0;
        for (int i = ti; i > 0; --i)
            if (c3[i] != 0.0) { d1 += c3[i] * (h.v2[i] - iv2); d2 += c3[i] * (h.v1[i] - iv1); }
        d1 += iv2 * m.intH3dash - iv2 * h.h3f; d2 += iv1 * m.intH3dash - iv1 * h.h3f;
        in1 += d1; in2 += d2;
    }
}

// ngspice ltraacld: Y0(jw) and exp(-lambda(jw) LEN) of the RLC, RC and LC lines (G = 0); the rows are
// Y0 v1 - i1 - Y0 e v2 - e i2 = 0 and likewise for port 2
struct AcTerms { std::complex<double> y0, e, y0e; };
inline AcTerms acTerms(const Model &m, double w) {
    double y0r = 0, y0i = 0, lr = 0, li = 0;
    if (m.kase == Case::LC) { y0r = m.admit; li = std::sqrt(m.L * m.C) * w; }
    else if (m.kase == Case::RLC) {
        double theta = 0.5 * std::atan(m.R / (w * m.L));
        double mag = std::sqrt(w * m.C / std::sqrt(m.R * m.R + w * w * m.L * m.L));
        y0r = mag * std::cos(theta); y0i = mag * std::sin(theta);
        theta = M_PI / 2 - theta;
        mag *= std::sqrt(m.R * m.R + w * w * m.L * m.L);
        lr = mag * std::cos(theta); li = mag * std::sin(theta);
    } else if (m.kase == Case::RC) {
        y0r = y0i = std::sqrt(0.5 * w * m.cByR);
        lr = li = std::sqrt(0.5 * w * m.R * m.C);
    }
    const std::complex<double> y0(y0r, y0i), e = std::exp(std::complex<double>(-lr * m.len, -li * m.len));
    return {y0, e, y0 * e};
}

// ngspice LTRAaccept after the point just pushed onto h: a breakpoint TD after the previous point when the
// travelling waves change (vntol: .options vntol; reltol / abstol: .options; the line's "rel" / "abs")
inline void acceptBreaks(const Model &m, History &h, double cktReltol, double cktAbstol, double vntol) {
    const size_t n = h.t.size();
    if (n < 2) return;
    const int k = static_cast<int>(n) - 1;
    auto w1 = [&](int i) { return (h.v1[i] + h.i1[i] * m.imped) * m.attenuation; };
    auto w2 = [&](int i) { return (h.v2[i] + h.i2[i] * m.imped) * m.attenuation; };
    auto brk = [&](double b) { h.breaks.insert(std::upper_bound(h.breaks.begin(), h.breaks.end(), b), b); };
    double v1 = w1(k), v2 = w1(k - 1), v3 = w2(k), v4 = w2(k - 1);
    if (std::fabs(v1 - v2) >= 50 * cktReltol * std::max(std::fabs(v1), std::fabs(v2)) + 50 * vntol ||
        std::fabs(v3 - v4) >= 50 * cktReltol * std::max(std::fabs(v3), std::fabs(v4)) + 50 * vntol)
        brk(h.t[k - 1] + m.td);
    v1 = w1(k); v2 = w1(k - 1); v3 = k < 2 ? v2 : w1(k - 2);
    double v4b = w2(k), v5 = w2(k - 1), v6 = k < 2 ? v5 : w2(k - 2);
    const double d1 = (v1 - v2) / (h.t[k] - h.t[k - 1]), d2 = k < 2 ? 0.0 : (v2 - v3) / (h.t[k - 1] - h.t[k - 2]);
    const double d3 = (v4b - v5) / (h.t[k] - h.t[k - 1]), d4 = k < 2 ? 0.0 : (v5 - v6) / (h.t[k - 1] - h.t[k - 2]);
    auto check = [&](double a, double b, double c) {
        return std::max({a, b, c}) - std::min({a, b, c}) >= std::fabs(50.0 * (cktReltol / 3.0 * (a + b + c) + cktAbstol));
    };
    if ((std::fabs(d1 - d2) >= m.reltol * std::max(std::fabs(d1), std::fabs(d2)) + m.abstol && check(v1, v2, v3)) ||
        (std::fabs(d3 - d4) >= m.reltol * std::max(std::fabs(d3), std::fabs(d4)) + m.abstol && check(v4b, v5, v6)))
        brk(h.t[k - 1] + m.td);
}

}  // namespace ltra
