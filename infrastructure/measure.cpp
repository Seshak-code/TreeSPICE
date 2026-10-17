// .meas evaluation (see engine_api/measure.h). Every vector is linear between its points, as ngspice reads it.
#include "engine_api/measure.h"
#include "engine_api/output.h"
#include "netlist/numparam.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <ostream>
#include <stdexcept>

namespace acutesim {
using compute::orchestration::MeasureResultDTO;
using compute::orchestration::SimulationResponseDTO;

namespace {

struct Fail : std::runtime_error { using std::runtime_error::runtime_error; };
const char* kOut = "out of interval";

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
std::string fmt(const char* f, double a, double b = 0.0) {
    char buf[128];
    std::snprintf(buf, sizeof buf, f, a, b);
    return buf;
}
bool number(const std::string& s, double& v) {
    try { v = numparam::eval(s, [](const std::string&, double&) { return false; }); return true; }
    catch (const std::exception&) { return false; }
}
double need(const std::string& s) {
    double v;
    if (!number(s, v)) throw Fail("'" + s + "' is not a number");
    return v;
}
std::string strip(std::string e) {
    if (e.size() >= 2 && (e.front() == '\'' || e.front() == '{' || e.front() == '"')) e = e.substr(1, e.size() - 2);
    return e;
}

// One measurement window of a vector: TRIG, TARG, WHEN or FIND ... WHEN: a level crossing.
struct Cross {
    std::string lhs, rhs;       // lhs = rhs; rhs a number or an expression
    int dir = 2, n = 1;         // 0 rise, 1 fall, 2 cross
    bool last = false, given = false;
    double td = 0.0;
};

struct Part {
    const SimulationResponseDTO& r;
    std::vector<double> x;
    std::map<std::string, std::vector<double>> cache;

    explicit Part(const SimulationResponseDTO& resp) : r(resp) { for (const auto& p : r.waveformPoints) x.push_back(p.timeS); }

    const std::vector<double>& vec(const std::string& e) {
        auto it = cache.find(e);
        if (it != cache.end()) return it->second;
        std::vector<double> v;
        const OutputTable t = selectOutputs(r, {e});
        for (const auto& c : t.columns[0].values) v.push_back(c.real());  // .ac: the real part, as ngspice
        return cache.emplace(e, std::move(v)).first->second;
    }
    std::vector<double> diff(const Cross& c) {  // lhs - rhs at every point
        std::vector<double> d = vec(c.lhs);
        double lv;
        if (number(c.rhs, lv)) for (auto& v : d) v -= lv;
        else { const auto& o = vec(c.rhs); for (size_t i = 0; i < d.size(); ++i) d[i] -= o[i]; }
        return d;
    }
    // The n-th (or last) crossing of d through 0 at or after td: the segment ending at index i, and the time.
    bool cross(const Cross& c, size_t& i, double& t) {
        const auto d = diff(c);
        int count = 0;
        bool found = false;
        for (size_t k = 1; k < x.size(); ++k) {
            const bool rise = d[k - 1] < 0 && d[k] >= 0, fall = d[k - 1] > 0 && d[k] <= 0;
            if (!((c.dir != 1 && rise) || (c.dir != 0 && fall))) continue;
            const double tk = x[k - 1] + (0 - d[k - 1]) * (x[k] - x[k - 1]) / (d[k] - d[k - 1]);
            if (tk < c.td) continue;
            found = true;
            i = k;
            t = tk;
            if (!c.last && ++count >= c.n) return true;
        }
        return found && c.last;
    }
    // y at x = t on segment i (linear)
    double at(const std::vector<double>& y, size_t i, double t) const {
        return y[i - 1] + (t - x[i - 1]) * (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
    }
    bool segment(double t, size_t& i) const {
        for (i = 1; i < x.size(); ++i) if (x[i - 1] <= t && t <= x[i]) return true;
        return false;
    }
};

// key=value options of a measure (val, td, rise, fall, cross, at, from, to) from tokens[from..to)
struct Opts {
    std::map<std::string, std::string> kv;
    std::string expr;  // the first token that is not key=value
    Opts(const std::vector<std::string>& t, size_t from, size_t to) {
        for (size_t i = from; i < to; ++i) {
            const size_t eq = t[i].find('=');
            const std::string key = lower(t[i].substr(0, eq));
            static const char* keys[] = {"val", "td", "rise", "fall", "cross", "at", "from", "to"};
            if (eq != std::string::npos && std::find(std::begin(keys), std::end(keys), key) != std::end(keys)) kv[key] = t[i].substr(eq + 1);
            else if (expr.empty()) expr = t[i];
            else throw Fail("bad syntax near '" + t[i] + "'");
        }
    }
    bool has(const char* k) const { return kv.count(k); }
    double num(const char* k, double dflt) const { return has(k) ? need(kv.at(k)) : dflt; }
    void edge(Cross& c) const {
        c.td = num("td", 0.0);
        for (const char* k : {"rise", "fall", "cross"}) {
            if (!has(k)) continue;
            c.dir = k[0] == 'r' ? 0 : k[0] == 'f' ? 1 : 2;
            c.given = true;
            c.last = lower(kv.at(k)) == "last";
            c.n = c.last ? 1 : std::max(1, static_cast<int>(need(kv.at(k))));
        }
    }
};

// "lhs=rhs" at the first '=' outside parentheses
Cross condition(const std::string& tok, const Opts& o) {
    Cross c;
    int depth = 0;
    for (size_t i = 0; i < tok.size(); ++i) {
        depth += (tok[i] == '(') - (tok[i] == ')');
        if (tok[i] == '=' && depth == 0) { c.lhs = tok.substr(0, i); c.rhs = tok.substr(i + 1); break; }
    }
    if (c.lhs.empty() || c.rhs.empty()) throw Fail("bad syntax. equal sign missing ?");
    o.edge(c);
    return c;
}

double crossTime(Part& p, const Cross& c, const char* where) {
    size_t i;
    double t;
    if (!p.cross(c, i, t)) throw Fail(std::string(where) + " : " + kOut);
    return t;
}

MeasureResultDTO run(Part& p, const std::vector<std::string>& t, const std::map<std::string, double>& done) {
    MeasureResultDTO m;
    m.name = lower(t[0]);
    const std::string kind = lower(t[1].substr(0, t[1].find('=')));
    auto why = [&](const char* label, const std::string& what) { throw Fail(std::string(label) + " : " + what); };
    if (kind == "param") {
        const size_t eq = t[1].find('=');
        const std::string e = strip(eq + 1 < t[1].size() ? t[1].substr(eq + 1) : t.size() > 2 ? t[2] : "");
        m.value = numparam::eval(e, [&](const std::string& n, double& v) {
            const auto it = done.find(lower(n));
            if (it == done.end()) return false;
            v = it->second;
            return true;
        });
        m.ok = true;
        return m;
    }
    if (kind == "trig") {
        size_t targ = 2;
        while (targ < t.size() && lower(t[targ]) != "targ") ++targ;
        if (targ == t.size()) why("trig(TARG)", "TARG is missing");
        double ts[2];
        for (int k = 0; k < 2; ++k) {
            const char* label = k ? "trig(TARG)" : "trig(TRIG)";
            const Opts o(t, k ? targ + 1 : 2, k ? t.size() : targ);
            if (o.has("at")) { ts[k] = o.num("at", 0); continue; }
            if (!o.has("val")) why(label, "at, rise, fall or cross must be given");
            Cross c;
            c.lhs = o.expr;
            c.rhs = o.kv.at("val");
            o.edge(c);
            if (!c.given) why(label, "at, rise, fall or cross must be given");
            ts[k] = crossTime(p, c, label);
        }
        m.value = ts[1] - ts[0];
        m.extra = " targ=" + fmt("%13.5e", ts[1]) + " trig=" + fmt("%13.5e", ts[0]);
        m.ok = true;
        return m;
    }
    if (kind == "when" || kind == "find" || kind == "deriv") {
        const bool plain = kind == "when";
        size_t w = 2 + !plain;  // index of AT= / WHEN / the condition
        if (t.size() <= w) throw Fail("bad syntax");
        const std::string expr = plain ? "" : t[2];
        const char* label = plain ? "when(WHEN)" : kind == "find" ? "find(AT)" : "deriv(AT)";
        size_t i;
        double at;
        if (!plain && lower(t[w]).rfind("at=", 0) == 0) {
            at = need(t[w].substr(3));
            if (!p.segment(at, i)) why(label, kOut);
        } else {
            if (!plain) { if (lower(t[w]) != "when") throw Fail("bad syntax. at or when expected"); ++w; }
            if (t.size() <= w) throw Fail("bad syntax");
            const Opts o(t, w + 1, t.size());
            if (!o.expr.empty()) throw Fail("bad syntax near '" + o.expr + "'");
            Cross c = condition(t[w], o);
            if (!c.given) { c.dir = 2; c.n = 1; }
            if (!p.cross(c, i, at)) why(label, kOut);
        }
        if (plain) m.value = at;
        else if (kind == "find") m.value = p.at(p.vec(expr), i, at);
        else m.value = (p.vec(expr)[i] - p.vec(expr)[i - 1]) / (p.x[i] - p.x[i - 1]);
        m.ok = true;
        return m;
    }
    // AVG RMS MIN MAX PP INTEG MAX_AT MIN_AT expr [FROM= TO=]
    static const char* stats[] = {"avg", "rms", "min", "max", "pp", "integ", "max_at", "min_at"};
    if (std::find(std::begin(stats), std::end(stats), kind) == std::end(stats)) throw Fail("unknown measurement '" + kind + "'");
    const Opts o(t, 2, t.size());
    if (o.expr.empty()) throw Fail("bad syntax. equal sign missing ?");
    const auto& y = p.vec(o.expr);
    // a window past the data is cut to it, as ngspice does (the transient's first point is after t=0)
    const double from = std::max(o.num("from", p.x.front()), p.x.front()), to = std::min(o.num("to", p.x.back()), p.x.back());
    const std::string label = kind + "(FROM/TO)";
    if (!(from <= to)) why(label.c_str(), kOut);
    if (kind == "avg" || kind == "rms" || kind == "integ") {
        if (!(from < to)) why(label.c_str(), kOut);
        // ponytail: trapezoids over the samples (rms: of y*y), so a coarse step costs accuracy; exact for piecewise-linear y
        double acc = 0.0, px = from, py = 0.0;
        size_t i;
        p.segment(from, i);
        py = p.at(y, i, from);
        const bool sq = kind == "rms";
        auto sqv = [&](double v) { return sq ? v * v : v; };
        for (size_t k = i; px < to; ++k) {
            const double nx = std::min(p.x[k], to), ny = nx == p.x[k] ? y[k] : p.at(y, k, nx);
            acc += 0.5 * (sqv(py) + sqv(ny)) * (nx - px);
            px = nx;
            py = ny;
        }
        m.value = kind == "integ" ? acc : sq ? std::sqrt(acc / (to - from)) : acc / (to - from);
        m.extra = " from=" + fmt("%13.5e", from) + " to=" + fmt("%13.5e", to);
        m.ok = true;
        return m;
    }
    size_t lo = SIZE_MAX, hi = 0;  // indices of the extremes among the samples in [from, to]
    bool any = false;
    for (size_t k = 0; k < p.x.size(); ++k) {
        if (p.x[k] < from || p.x[k] > to) continue;
        if (!any) lo = hi = k;
        any = true;
        if (y[k] < y[lo]) lo = k;
        if (y[k] > y[hi]) hi = k;
    }
    if (!any) why(label.c_str(), kOut);
    if (kind == "min") { m.value = y[lo]; m.extra = " at=" + fmt("%13.5e", p.x[lo]); }
    else if (kind == "max") { m.value = y[hi]; m.extra = " at=" + fmt("%13.5e", p.x[hi]); }
    else if (kind == "pp") { m.value = y[hi] - y[lo]; m.extra = " from=" + fmt("%13.5e", from) + " to=" + fmt("%13.5e", to); }
    else { const size_t k = kind == "max_at" ? hi : lo; m.value = p.x[k]; m.extra = " with=" + fmt("%13.5e", y[k]); }
    m.ok = true;
    return m;
}

}  // namespace

void attachMeasures(SimulationResponseDTO& r) {
    const std::string an = analysisName(r);
    std::vector<const compute::orchestration::OutputRequestDTO*> cards;
    r.measures.clear();
    for (const auto& q : r.outputRequests) {
        if (q.kind != 'm') continue;
        if (q.analysis == an) cards.push_back(&q);
        else if (q.analysis != "tran" && q.analysis != "dc" && q.analysis != "ac") {  // ngspice reports it and goes on
            MeasureResultDTO m;
            m.name = lower(q.specs[0]);
            m.error = "Error: unrecognized analysis type '" + q.analysis + "' for the following .meas statement: " + m.name + "\n";
            r.measures.push_back(std::move(m));
        }
    }
    if (cards.empty() && r.measures.empty()) return;
    int step = 0;
    for (const auto& sub : splitByOuter(r)) {
        Part part(sub);
        std::map<std::string, double> done;
        for (const auto* q : cards) {
            std::vector<std::string> t;  // "val = 0.5" -> "val=0.5"
            for (const auto& w : q->specs)
                if (!t.empty() && (w[0] == '=' || t.back().back() == '=')) t.back() += w;
                else t.push_back(w);
            MeasureResultDTO m;
            m.name = lower(t[0]);
            try {
                if (t.size() < 2 || part.x.size() < 2) throw Fail(kOut);
                m = run(part, t, done);
                done[m.name] = m.value;
            } catch (const std::exception& e) {
                std::string echo = ".meas " + an;
                for (const auto& w : t) echo += " " + lower(w);
                m.ok = false;
                const std::string what = e.what();
                m.error = what.find(" : ") != std::string::npos ? "Error: measure  " + m.name + "  " + what + "\n " + echo + " failed!\n"
                                                                : "Error: measure  " + m.name + " failed:\n\t" + what + "\n " + echo + " failed!\n";
            }
            m.step = step;
            r.measures.push_back(std::move(m));
        }
        ++step;
    }
}

void writeMeasures(std::ostream& os, const SimulationResponseDTO& r, int step) {
    bool header = false;
    for (const auto& m : r.measures) {
        if (m.step != step) continue;
        if (!header) {
            const std::string an = analysisName(r);
            os << "  Measurements for " << (an == "tran" ? "Transient" : an == "ac" ? "AC" : "DC") << " Analysis\n\n";
            header = true;
        }
        if (!m.ok) { os << m.error << "\n"; continue; }
        char b[64];
        std::snprintf(b, sizeof b, "%-20s=%14.5e", m.name.c_str(), m.value);
        os << b << m.extra << "\n";
    }
    if (header) os << "\n";
}

}  // namespace acutesim
