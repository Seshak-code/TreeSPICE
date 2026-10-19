#pragma once
// XSPICE event-driven digital simulation, as ngspice 47 runs its code models (behaviour probed against
// ngspice 47; references in tests/ngspice_diff/netlists/x_xspice_*). A digital node holds one of 12
// states, 0/1/U at strong, resistive, hi-Z or undetermined strength. An instance evaluates when an input
// changes and posts output events after its delays. adc_bridge samples analog nodes at accepted time
// points; dac_bridge ramps an analog voltage source the parser added.
#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace xspice {

enum Lv : uint8_t { L0, L1, LU };
enum St : uint8_t { SS, SR, SZ, SU };  // strong, resistive, hi-impedance, undetermined
struct Val {
    Lv v = LU;
    St s = SZ;
    bool operator==(const Val &o) const { return v == o.v && s == o.s; }
    bool operator!=(const Val &o) const { return !(*this == o); }
};
inline std::string text(Val x) { return std::string(1, "01U"[x.v]) + "srzu"[x.s]; }

// Two drivers on one node: all 144 pairs probed in ngspice 47. Not associative (1r, 0z, 1u gives 1u in
// that order, Uu in another): a node folds its drivers in netlist order, as ngspice does.
inline Val resolve(Val a, Val b) {
    static const char *T[12] = {  // row a, column b; index strength * 3 + level, order 0s 1s Us 0r ... Uu
        "0sUsUs0s0s0s0s0s0s0sUsUs", "Us1sUs1s1s1s1s1s1sUs1sUs", "UsUsUsUsUsUsUsUsUsUsUsUs",
        "0s1sUs0rUrUr0r0r0r0uUuUu", "0s1sUsUr1rUr1r1r1rUu1uUu", "0s1sUsUrUrUrUrUrUrUuUuUu",
        "0s1sUs0r1rUr0zUzUz0uUuUu", "0s1sUs0r1rUrUz1zUzUu1uUu", "0s1sUs0r1rUrUzUzUzUuUuUu",
        "0sUsUs0uUuUu0uUuUu0uUuUu", "Us1sUsUu1uUuUu1uUuUu1uUu", "UsUsUsUuUuUuUuUuUuUuUuUu"};
    const char *c = T[a.s * 3 + a.v] + 2 * (b.s * 3 + b.v);
    return {c[0] == '0' ? L0 : c[0] == '1' ? L1 : LU, c[1] == 's' ? SS : c[1] == 'r' ? SR : c[1] == 'z' ? SZ : SU};
}

enum class Kind { BUF, INV, AND, NAND, OR, NOR, XOR, XNOR, TRI, PULLUP, PULLDOWN, DFF, JKFF, TFF, SRFF, DLATCH, SRLATCH, OPENC, OPENE, FDIV, LUT, SOURCE, OSC, PWM, ADC, DAC };
struct Port { int node = -1; bool inv = false; };  // a digital node, or -1 (NULL)
struct Inst {
    std::string name;
    Kind kind = Kind::BUF;
    std::vector<Port> in, out;  // digital; flip-flops / latches in: data (JK, SR: two) clk (latch: enable) set reset, out: out Nout; tristate in: in enable
    std::vector<int> analog;    // adc_bridge inputs / dac_bridge outputs (global analog nodes)
    std::vector<int> vsrc;      // dac_bridge: its voltage source per output (global block)
    double rise = 1e-9, fall = 1e-9;              // output delays (tristate: its delay, both; open collector / emitter: open_delay is the open edge)
    double clk = 1e-9, set = 1e-9, reset = 1e-9;  // flip-flops (latch: clk is enable_delay): added to rise / fall
    double data = 1e-9;                           // latches: data_delay / sr_delay
    int div = 2, high = 1, icount = 0;            // d_fdiv: div_factor, high_cycles, i_count
    std::vector<Lv> table;                        // d_lut: the output per input word (in[0] the low bit)
    std::vector<double> times;                    // d_source: each row's time (increasing) and its outputs
    std::vector<std::vector<Val>> rows;
    std::vector<double> ctl, freq;                // d_osc / d_pwm: the control table (ctl increasing) to frequency / duty
    double duty = 0.5, phase = 0.0;               // d_osc: duty_cycle; both: init_phase (degrees)
    double period = 1e-6;                         // d_pwm: 1 / frequency
    Lv ic = L0;
    bool inertial = false;                        // inertial_delay (buffer, inverter, gates): a pulse shorter than the delay is dropped
    double lo = 0.1, hi = 0.9;                                             // adc_bridge in_low / in_high
    double vlo = 0.0, vhi = 1.0, vun = 0.5, trise = 1e-9, tfall = 1e-9;    // dac_bridge
};
struct Net {
    std::vector<std::string> nodes;  // digital node names
    std::vector<Inst> insts;
};

class Sim {
public:
    struct Change { double t; int node; Val v; };  // a node's resolved value
    struct Ramp { double t; int vsrc; double target, riseSlope, fallSlope; };  // a dac_bridge output from t

    explicit Sim(std::shared_ptr<const Net> net) : net_(std::move(net)) {
        const auto &n = *net_;
        node_.assign(n.nodes.size(), Val{});
        drivers_.resize(n.nodes.size());
        fanout_.resize(n.nodes.size());
        for (size_t i = 0; i < n.insts.size(); ++i) {
            const auto &x = n.insts[i];
            for (size_t p = 0; p < x.out.size(); ++p)
                if (x.out[p].node >= 0) drivers_[x.out[p].node].push_back({(int)i, (int)p});
            for (const auto &p : x.in)
                if (p.node >= 0 && (fanout_[p.node].empty() || fanout_[p.node].back() != (int)i)) fanout_[p.node].push_back((int)i);
        }
        st_.resize(n.insts.size());
        for (size_t i = 0; i < n.insts.size(); ++i) {
            const auto &x = n.insts[i];
            st_[i].drive.assign(x.out.size(), Val{});
            st_[i].last.assign(x.out.size(), Val{});
            st_[i].prev.assign(x.in.size(), LU);
            st_[i].when.assign(x.out.size(), -1.0);
            st_[i].iprev.assign(x.out.size(), LU);
            st_[i].adc.assign(x.kind == Kind::ADC ? x.analog.size() : 0, static_cast<Lv>(3));  // none yet: the first sample posts, U too
            st_[i].dac.assign(x.kind == Kind::DAC ? x.analog.size() : 0, -1e300);
        }
    }

    // t = 0: settle with zero delays, the adc_bridges reading v. Returns each dac_bridge output's DC value.
    // The caller solves the analog circuit with them and calls again until adcChanged(v) is false.
    std::vector<std::pair<int, double>> settle(const std::function<double(int)> &v) {
        const auto &n = *net_;
        for (size_t i = 0; i < n.insts.size(); ++i)
            if (n.insts[i].kind == Kind::ADC) sampleAdc((int)i, 0.0, v, true);
            else if (n.insts[i].kind == Kind::OSC || n.insts[i].kind == Kind::PWM) oscStart((int)i, v);
        for (int pass = 0; pass < 200; ++pass) {  // event passes until nothing changes
            bool any = false;
            for (size_t i = 0; i < n.insts.size(); ++i) any |= eval((int)i, 0.0, true);
            if (!any) break;
        }
        std::vector<std::pair<int, double>> dc;
        for (size_t i = 0; i < n.insts.size(); ++i)
            if (n.insts[i].kind == Kind::DAC)
                for (size_t k = 0; k < n.insts[i].analog.size(); ++k) {
                    st_[i].dac[k] = dacTarget(n.insts[i], level(n.insts[i].in[k]));
                    dc.push_back({n.insts[i].vsrc[k], st_[i].dac[k]});
                }
        for (size_t i = 0; i < n.insts.size(); ++i)
            if (n.insts[i].kind == Kind::SOURCE) sourceRow((int)i);
        log_.clear();
        for (size_t k = 0; k < node_.size(); ++k) log_.push_back({0.0, (int)k, node_[k]});
        return dc;
    }
    bool adcChanged(const std::function<double(int)> &v) const {
        const auto &n = *net_;
        for (size_t i = 0; i < n.insts.size(); ++i)
            if (n.insts[i].kind == Kind::ADC)
                for (size_t k = 0; k < n.insts[i].analog.size(); ++k)
                    if (adcLevel(n.insts[i], v(n.insts[i].analog[k])) != st_[i].adc[k]) return true;
        return false;
    }

    // The first pending event after x (infinity when none): the transient lands a step on it
    double next(double x) const {
        double best = 1e300;
        for (const auto &[t, e] : queue_) if (t > x) { best = t; break; }
        for (size_t i = 0; i < st_.size(); ++i)
            if ((net_->insts[i].kind == Kind::OSC || net_->insts[i].kind == Kind::PWM) && st_[i].wake > x) best = std::min(best, st_[i].wake);
        return best;
    }

    // An accepted time point t: process the events up to t, then sample the adc_bridges at t.
    // Returns the dac_bridge ramps that start (at their event's time).
    std::vector<Ramp> advance(double t, double tol, const std::function<double(int)> &v) {
        ramps_.clear();
        while (!queue_.empty() && queue_.begin()->first <= t + tol) {
            const double te = queue_.begin()->first;
            std::vector<int> touched;
            while (!queue_.empty() && queue_.begin()->first == te) {
                const Ev e = queue_.begin()->second;
                queue_.erase(queue_.begin());
                st_[e.inst].drive[e.port] = e.v;
                touched.push_back(net_->insts[e.inst].out[e.port].node);
                if (net_->insts[e.inst].kind == Kind::SOURCE && e.port == 0) { ++st_[e.inst].count; sourceRow(e.inst); }
            }
            propagate(touched, te, false);
        }
        const auto &n = *net_;
        for (size_t i = 0; i < n.insts.size(); ++i)
            if (n.insts[i].kind == Kind::ADC) sampleAdc((int)i, t, v, false);
            else if (n.insts[i].kind == Kind::OSC || n.insts[i].kind == Kind::PWM) oscCall((int)i, t, v, false);
        return ramps_;
    }

    const std::vector<Change> &log() const { return log_; }
    const Net &net() const { return *net_; }

private:
    struct Ev { int inst, port; Val v; };
    struct State {
        std::vector<Val> drive, last;  // the value each output drives now / the last one computed (ngspice out_old)
        std::vector<Lv> prev;          // inputs at the last evaluation (d_dff edges)
        std::vector<Lv> adc;           // adc_bridge: the last level computed per channel
        std::vector<double> dac;       // dac_bridge: the voltage each output heads for
        std::vector<double> when;      // inertial: the pending change's time per output (<= now: none)
        std::vector<Lv> iprev;         // inertial: the output's value before that change
        int count = 0;                 // d_fdiv: the edge count; d_source: the next row
        Lv osc = L0;                   // d_osc: its output (ngspice state->last), set at oscT; its next call at wake
        double oscT = 0.0, wake = -1.0;
    };
    std::shared_ptr<const Net> net_;
    std::vector<Val> node_;
    std::vector<std::vector<std::pair<int, int>>> drivers_;  // per node: (instance, output), netlist order
    std::vector<std::vector<int>> fanout_;                    // per node: the instances reading it
    std::vector<State> st_;
    std::multimap<double, Ev> queue_;
    std::vector<Change> log_;
    std::vector<Ramp> ramps_;

    Lv level(const Port &p) const {
        if (p.node < 0) return L0;  // NULL reads as 0 (d_dff set/reset)
        const Lv l = node_[p.node].v;
        return p.inv && l != LU ? (l == L0 ? L1 : L0) : l;
    }
    static Lv invert(Lv l) { return l == LU ? LU : l == L0 ? L1 : L0; }
    static Lv adcLevel(const Inst &x, double v) { return v >= x.hi ? L1 : v <= x.lo ? L0 : LU; }
    static double dacTarget(const Inst &x, Lv l) { return l == L1 ? x.vhi : l == L0 ? x.vlo : x.vun; }

    // Re-resolve the touched nodes; log the ones that changed and evaluate what reads them
    void propagate(std::vector<int> touched, double t, bool dc) {
        std::sort(touched.begin(), touched.end());
        touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
        std::vector<int> evalList;
        for (int k : touched) {
            if (k < 0) continue;
            Val r{};
            bool first = true;
            for (const auto &[i, p] : drivers_[k]) {
                const Val d = st_[i].drive[p];
                r = first ? d : resolve(r, d);
                first = false;
            }
            if (r == node_[k]) continue;
            node_[k] = r;
            if (!dc) log_.push_back({t, k, r});
            evalList.insert(evalList.end(), fanout_[k].begin(), fanout_[k].end());
        }
        std::sort(evalList.begin(), evalList.end());
        evalList.erase(std::unique(evalList.begin(), evalList.end()), evalList.end());
        for (int i : evalList) eval(i, t, dc);
    }

    // Queue output p's value v at te (ngspice EVTqueue_output: it cancels the output's pending events from te on).
    // An inverted output (~out) drives the inverse.
    void queue(int i, int p, Val v, double te) {
        if (net_->insts[i].out[p].inv) v.v = invert(v.v);
        for (auto it = queue_.begin(); it != queue_.end();)
            it = it->first >= te && it->second.inst == i && it->second.port == p ? queue_.erase(it) : std::next(it);
        queue_.insert({te, {i, p, v}});
    }
    // Post output p's new value. At DC it applies at once. Returns whether the computed output changed.
    // Inertial (ngspice's buffer / gate code models): a change while one is pending either reverts it, the
    // output scheduled back at half the remaining delay (a glitch shorter than the delay vanishes), or, to a third
    // value, restores the old value there and then posts the new one with its own delay.
    bool post(int i, int p, Val v, double t, double delay, bool dc) {
        auto &s = st_[i];
        if (v == s.last[p]) return false;
        const Val old = s.last[p];
        s.last[p] = v;
        const auto &x = net_->insts[i];
        if (dc) {
            s.drive[p] = x.out[p].inv ? Val{invert(v.v), v.s} : v;
            propagate({x.out[p].node}, 0.0, true);
            return true;
        }
        if (x.inertial) {
            if (s.when[p] <= t) { s.iprev[p] = old.v; s.when[p] = t + delay; }
            else if (v.v != s.iprev[p]) {
                queue(i, p, {s.iprev[p], drive(x, s.iprev[p])}, t + (s.when[p] - t) / 2.0);
                if (v.v == LU) delay = s.iprev[p] == L0 ? x.rise : x.fall;
                s.when[p] = t + delay;
            } else {
                delay = (s.when[p] - t) / 2.0;
                s.when[p] = -1.0;
            }
        }
        queue(i, p, v, t + std::max(delay, 1e-12));  // a zero delay is 1 ps in ngspice
        return true;
    }
    // d_source: queue its next row (one at a time, as ngspice, so a long file does not fill the queue)
    void sourceRow(int i) {
        const auto &x = net_->insts[i];
        const size_t k = st_[i].count;
        if (k >= x.times.size()) return;
        for (size_t p = 0; p < x.out.size(); ++p) queue(i, (int)p, x.rows[k][p], x.times[k]);
    }
    // An output's strength at level l: d_open_c drives only 0, d_open_e only 1 (the other level is high impedance)
    static St drive(const Inst &x, Lv l) {
        if (l == LU) return x.kind == Kind::OPENC || x.kind == Kind::OPENE ? SU : SS;
        return x.kind == Kind::OPENC && l == L1 ? SZ : x.kind == Kind::OPENE && l == L0 ? SZ : SS;
    }
    // The flip-flops' and latches' t = 0 set / reset: each wins only with the other at 0
    static Lv asyncDc(Lv set, Lv rst, Lv q) {
        return set == L1 && rst == L1 ? LU : set == L1 && rst == L0 ? L1 : set == L0 && rst == L1 ? L0 : q;
    }
    // ngspice's cm_eval_jk_result / cm_eval_sr_result: 00 holds, 01 resets, 10 sets, 11 toggles (JK) or is U (SR); U in gives U
    static Lv jkSr(Lv a, Lv b, Lv old, bool jk) {
        if (a == LU || b == LU) return LU;
        if (a == L0) return b == L0 ? old : L0;
        return b == L0 ? L1 : jk ? invert(old) : LU;
    }
    // ngspice's delay choice: rise to 1, fall to 0; to U, rise from 0 and fall from 1
    double edge(const Inst &x, Val old, Lv now, double base) const {
        const bool up = now == L1 || (now == LU && old.v == L0);
        return base + (up ? x.rise : x.fall);
    }

    bool eval(int i, double t, bool dc) {
        const auto &x = net_->insts[i];
        auto &s = st_[i];
        std::vector<Lv> in(x.in.size());
        for (size_t k = 0; k < x.in.size(); ++k) in[k] = level(x.in[k]);
        bool changed = false;
        auto out = [&](int p, Lv l, St strength, double base) {
            changed |= post(i, p, {l, strength}, t, edge(x, s.last[p], l, base), dc);
        };
        auto any = [&](Lv l) { return std::find(in.begin(), in.end(), l) != in.end(); };
        switch (x.kind) {
        case Kind::BUF: out(0, in[0], SS, 0); break;
        case Kind::INV: out(0, invert(in[0]), SS, 0); break;
        case Kind::AND: case Kind::NAND: {
            const Lv r = any(L0) ? L0 : any(LU) ? LU : L1;
            out(0, x.kind == Kind::AND ? r : invert(r), SS, 0);
            break;
        }
        case Kind::OR: case Kind::NOR: {
            const Lv r = any(L1) ? L1 : any(LU) ? LU : L0;
            out(0, x.kind == Kind::OR ? r : invert(r), SS, 0);
            break;
        }
        case Kind::XOR: case Kind::XNOR: {
            Lv r = L0;
            if (any(LU)) r = LU;
            else for (Lv l : in) r = l == L1 ? invert(r) : r;
            out(0, x.kind == Kind::XOR ? r : invert(r), SS, 0);
            break;
        }
        case Kind::TRI: out(0, in[0], in[1] == L1 ? SS : in[1] == L0 ? SZ : SU, 0); break;
        case Kind::OPENC: case Kind::OPENE: out(0, in[0], drive(x, in[0]), 0); break;
        case Kind::FDIV: {
            // ngspice d_fdiv: a rising input (to 1 from 0 or U) counts; at div_factor + 1 the count wraps to 1 and the
            // output rises, at high_cycles + 1 it falls. t = 0: i_count outside [0, div_factor) is 0; the output is 1
            // for a count in [1, high_cycles], else 0
            Lv q = s.last[0].v;
            if (dc) {
                s.count = x.icount >= 0 && x.icount < x.div ? x.icount : 0;
                q = s.count > 0 && s.count <= x.high ? L1 : L0;
            } else if (in[0] != s.prev[0] && in[0] == L1) {
                if (++s.count == x.div + 1 || s.count == 1) { s.count = 1; q = L1; }
                else if (s.count == x.high + 1) q = L0;
            }
            out(0, q, SS, 0);
            break;
        }
        case Kind::SOURCE:  // ngspice d_source: a row at t = 0 is the DC value, else U; the rest are queued (sourceRow)
            if (dc) {
                s.count = !x.times.empty() && x.times[0] == 0.0 ? 1 : 0;
                for (size_t p = 0; p < x.out.size(); ++p) changed |= post(i, (int)p, s.count ? x.rows[0][p] : Val{LU, SU}, t, 0.0, true);
            }
            break;
        case Kind::LUT: {
            size_t idx = 0;
            for (size_t k = 0; k < in.size(); ++k) idx |= size_t{in[k] == L1} << k;
            out(0, any(LU) ? LU : x.table[idx], SS, 0);
            break;
        }
        case Kind::PULLUP: out(0, L1, SR, 0); break;
        case Kind::PULLDOWN: out(0, L0, SR, 0); break;
        case Kind::DFF: case Kind::JKFF: case Kind::TFF: case Kind::SRFF: {
            // ngspice d_dff / d_jkff / d_tff / d_srff (one set / reset logic): a set or reset change acts at once;
            // otherwise a rising clock (to 1 from 0 or U) with neither set nor reset at 1 takes the next state.
            // Set and reset at U keep an output that agrees.
            const size_t nd = x.kind == Kind::JKFF || x.kind == Kind::SRFF ? 2 : 1;
            const Lv clk = in[nd], set = in[nd + 1], rst = in[nd + 2];
            const Lv old = s.last[0].v;
            Lv q = old;
            double base = 0.0;
            if (dc) q = asyncDc(set, rst, x.ic);
            else if (set != s.prev[nd + 1]) {
                base = x.set;
                if (set == L1) q = rst == L1 ? LU : L1;
                else if (set == L0) q = rst == L1 ? L0 : old;
                else q = rst == L1 ? (old == L0 ? L0 : LU) : (old == L1 ? L1 : LU);
            } else if (rst != s.prev[nd + 2]) {
                base = x.reset;
                if (rst == L1) q = set == L1 ? LU : L0;
                else if (rst == L0) q = set == L1 ? L1 : old;
                else q = set == L1 ? (old == L1 ? L1 : LU) : (old == L0 ? L0 : LU);
            } else if (clk != s.prev[nd] && clk == L1 && set != L1 && rst != L1) {
                base = x.clk;
                const Lv a = in[0], b = nd > 1 ? in[1] : LU;
                if (x.kind == Kind::DFF) q = a;
                else if (x.kind == Kind::TFF) q = a == L1 ? invert(old) : a == L0 ? old : LU;
                else q = jkSr(a, b, old, x.kind == Kind::JKFF);
            }
            out(0, q, SS, base);
            out(1, invert(q), SS, base);
            break;
        }
        case Kind::DLATCH: case Kind::SRLATCH: {
            // ngspice d_dlatch / d_srlatch: a set change acts, else a reset change, else an enable change, else a data change.
            // Releasing set or reset (to 0 or U) while enabled takes the data at once; set and reset at U count as 0.
            const bool sr = x.kind == Kind::SRLATCH;
            const size_t nd = sr ? 2 : 1;
            const Lv en = in[nd], set = in[nd + 1], rst = in[nd + 2];
            const Lv old = s.last[0].v;
            const Lv d = sr ? jkSr(in[0], in[1], old, false) : in[0];
            Lv q = old;
            double base = 0.0;
            if (dc) {
                if (en == L1) q = !sr ? in[0] : in[0] == L1 && in[1] == L0 ? L1 : in[0] == L0 && in[1] == L1 ? L0 : x.ic;
                else q = x.ic;
                q = asyncDc(set, rst, q);
            } else if (set != s.prev[nd + 1]) {
                base = x.set;
                q = set == L1 ? (rst == L1 ? LU : L1) : rst == L1 ? L0 : en == L1 ? d : old;
            } else if (rst != s.prev[nd + 2]) {
                base = x.reset;
                q = rst == L1 ? (set == L1 ? LU : L0) : set == L1 ? L1 : en == L1 ? d : old;
            } else if (set != L1 && rst != L1 && en != s.prev[nd]) {
                base = x.clk;
                q = en == L1 ? d : old;
            } else if (set != L1 && rst != L1 && en == L1 && (in[0] != s.prev[0] || (sr && in[1] != s.prev[1]))) {
                base = x.data;
                q = d;
            }
            out(0, q, SS, base);
            out(1, invert(q), SS, base);
            break;
        }
        case Kind::DAC:
            for (size_t k = 0; k < x.analog.size(); ++k) {
                const double target = dacTarget(x, in[k]);
                if (target == s.dac[k]) continue;
                s.dac[k] = target;
                if (!dc) ramps_.push_back({t, x.vsrc[k], target, (x.vhi - x.vlo) / x.trise, (x.vhi - x.vlo) / x.tfall});
                changed = true;
            }
            break;
        case Kind::ADC: case Kind::OSC: case Kind::PWM: break;  // called at time points (sampleAdc, oscCall)
        }
        s.prev = in;
        return changed;
    }

    // ngspice d_osc's period (and its fixed duty) or d_pwm's fixed period and duty at control c: the table's
    // piecewise-linear value, its end segments extended (d_pwm's duty then held to [0.01, 0.99])
    static std::pair<double, double> oscPD(const Inst &x, double c) {
        const size_t n = x.ctl.size();
        size_t k = 0;
        while (k < n && x.ctl[k] <= c) ++k;
        if (k > 0) k = k == n ? n - 2 : k - 1;
        const double y = x.freq[k] + (c - x.ctl[k]) * ((x.freq[k + 1] - x.freq[k]) / (x.ctl[k + 1] - x.ctl[k]));
        if (x.kind == Kind::PWM) return {x.period, std::clamp(y, 0.01, 0.99)};
        if (!(y > 0.0)) throw std::runtime_error(x.name + ": d_osc control " + std::to_string(c) + " V gives a frequency <= 0");
        return {1.0 / y, x.duty};
    }
    // t = 0: the state a transition before t = 0 would have left (init_phase into the cycle, which starts low), then
    // the t = 0 call
    void oscStart(int i, const std::function<double(int)> &v) {
        const auto &x = net_->insts[i];
        auto &s = st_[i];
        const auto [P, d] = oscPD(x, v(x.analog[0]));
        double ph = x.phase / 360.0;
        if (ph < 0.0) ph += 1.0;
        s.oscT = P * (1.0 - d - ph);
        if (s.oscT < 0.0) s.osc = L1;
        else { s.osc = L0; s.oscT = -P * ph; }
        s.wake = -1.0;
        oscCall(i, 0.0, v, true);
    }
    // ngspice cm_d_osc / cm_d_pwm's event call (each accepted point and its own requests): once 3/4 of the half
    // period has passed, post the transition at its time from the present period, and ask for a call 4/5 into the next half
    void oscCall(int i, double t, const std::function<double(int)> &v, bool dc) {
        const auto &x = net_->insts[i];
        auto &s = st_[i];
        const auto [P, d] = oscPD(x, v(x.analog[0]));
        double delta = P * (s.osc == L1 ? d : 1.0 - d), when = s.oscT + delta;
        // ponytail: a missed transition (the control raised the frequency within one step) makes ngspice reject
        // the step and land just before it; the event loop cannot reject steps, so it posts at t, up to a step late
        if (t >= when) when = t;
        if (t >= s.oscT + 0.75 * delta) {
            s.oscT = when;
            s.osc = invert(s.osc);
            post(i, 0, {s.osc, SS}, t, when - t, dc);
            // d_pwm scales the half it just ended by the next one's share (ngspice's delta *= ddc): its request comes
            // early, and the call then asks again; kept, as the extra call lands an analog step
            delta = (x.kind == Kind::PWM ? delta : P) * (s.osc == L1 ? d : 1.0 - d);
            s.wake = s.oscT + 0.8 * delta;
        } else {
            // ngspice's t = 0 call sets the output but then marks it unchanged: the node keeps XSPICE's initial 0u
            if (dc) post(i, 0, {L0, SU}, t, 0.0, true);
            if (t >= s.oscT) s.wake = s.oscT + 0.8 * delta;
        }
    }

    void sampleAdc(int i, double t, const std::function<double(int)> &v, bool dc) {
        const auto &x = net_->insts[i];
        auto &s = st_[i];
        for (size_t k = 0; k < x.analog.size(); ++k) {
            const Lv l = adcLevel(x, v(x.analog[k]));
            if (l == s.adc[k]) continue;
            s.adc[k] = l;
            post(i, (int)k, {l, SS}, t, edge(x, s.last[k], l, 0.0), dc);
        }
    }
};

}  // namespace xspice
