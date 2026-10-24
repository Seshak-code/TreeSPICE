#pragma once
// S3.1 (docs/scale-s3.md): the classes of an unflattened netlist's nets (TensorNetlist::hier) on which its solution
// is constant. Color refinement (one-dimensional Weisfeiler-Leman) of each definition in each context it is
// instantiated in: a context is a definition with the colors of its pins seen from its parent. Inside a context a
// net's color is refined by the colors of the elements it touches (at which terminal) and of the child instances it
// is a pin of; a child instance is (its definition, the colors of its port nets), so two children with equal port
// colors are one context, refined once. The coarsest such fixed point is an equitable partition: every net of a
// class meets the same elements and children, class for class, so the MNA solution (unique, or Newton's from a
// class-constant start) is constant on each class and solves the quotient circuit.
//
// Colors are exact: interned descriptors, never hashes (a collision would merge nets that differ). Elements are
// equal when they are the same element of the same definition, or R, C, L, V, I with every field equal (the
// testbench's identical drivers). Couplings that do not run through terminals (B/E/G expressions, F/H/W control
// currents, K, XSPICE) are not modelled yet: a netlist with one is refused, never partitioned wrongly.
#include "circuit.h"

#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

namespace hierpart {

// An aggregate's field count (brace initialization from convertible placeholders): the R/C/L/V/I descriptors below
// list every field, and this fails the build when one is added (a size check would not: library-dependent sizes)
struct AnyField { template <class T> operator T() const; };
template <class T, class... A> constexpr auto initable(int) -> decltype(T{std::declval<A>()...}, true) { return true; }
template <class T, class... A> constexpr bool initable(...) { return false; }
template <class T, class... A> constexpr size_t fields() {
    if constexpr (initable<T, A..., AnyField>(0)) return fields<T, A..., AnyField>();
    else return sizeof...(A);
}

struct Context {
    std::string def;          // "": the top level
    std::vector<int> pins;    // its pins' colors (the context's identity)
    std::vector<int> net;     // each local node's color (0: ground; the top level: global nodes)
    std::vector<int> child;   // each child instance's context (hier.children[def], hier.top)
};

struct Partition {
    std::vector<Context> ctx;  // ctx[0]: the top level
    // net classes: distinct colors of each context's nodes (ground excluded), summed over contexts
    size_t netClasses() const {
        size_t n = 0;
        for (const auto& c : ctx) {
            std::vector<int> col(c.net.begin() + (c.net.empty() ? 0 : 1), c.net.end());
            std::sort(col.begin(), col.end());
            n += (size_t)(std::unique(col.begin(), col.end()) - col.begin());
        }
        return n;
    }
};

class Refiner {
public:
    explicit Refiner(const TensorNetlist& nl) : nl_(nl) {}

    Partition run() {
        Partition p;
        p_ = &p;
        Context top;
        top.def = "";
        refine(nl_.globalBlock, nl_.numGlobalNodes, top.pins, topSeeds(), nl_.hier.top, top);
        p.ctx.push_back(top);
        for (size_t k = 0; k < nl_.hier.top.size(); ++k) p.ctx[0].child[k] = context(nl_.hier.top[k].def, pinsOf(nl_.hier.top[k], p.ctx[0].net));
        return p;
    }

private:
    using Key = std::vector<long long>;
    const TensorNetlist& nl_;
    Partition* p_ = nullptr;
    std::map<Key, int> colors_;                      // descriptor -> color
    std::map<std::pair<std::string, std::vector<int>>, int> contexts_;  // (definition, pin colors) -> context

    std::map<std::string, int> strs_;                // string -> id (exact)
    int color(const Key& k) { return colors_.emplace(k, (int)colors_.size()).first->second; }
    int strId(const std::string& s) { return strs_.emplace(s, (int)strs_.size()).first->second; }
    static long long bits(double v) { long long b; std::memcpy(&b, &v, sizeof b); return b; }
    static void add(Key& k, const std::vector<double>& v) { k.push_back((long long)v.size()); for (double x : v) k.push_back(bits(x)); }

    struct El { int id; std::vector<int> t; };

    // The block's elements with their identity and ordered terminals; refuses what the refinement does not model
    std::vector<El> elements(const TensorBlock& b, const std::string& def) {
        const std::string where = def.empty() ? "the top level" : def;
        if (!b.behavioural.empty() || !b.currentControlledVoltageSources.empty() || !b.currentControlledCurrentSources.empty() ||
            !b.mutualInductors.empty())
            throw std::runtime_error(where + ": B/E/G expressions, F/H and K are not supported unflattened yet");
        for (const auto& s : b.switches)
            if (!s.ctrlName.empty()) throw std::runtime_error(where + ": W switches are not supported unflattened yet");
        std::vector<El> out;
        const int d = strId(def);
        auto byIndex = [&](char kind, size_t i, std::vector<int> t) { out.push_back({color({1, kind, d, (long long)i}), std::move(t)}); };
        // R, C, L, V, I by value: every field (a static_assert fails when one is added, so this list is kept whole)
        static_assert(fields<Resistor>() == 16 && fields<Capacitor>() == 17 && fields<Inductor>() == 12 &&
                          fields<VoltageSource>() == 26 && fields<CurrentSource>() == 26,
                      "an R/C/L/V/I struct gained or lost a field: update its descriptor in hier_partition.h (and the count here)");
        for (const auto& r : b.resistors) {
            Key k{2, 'R', strId(r.model_name), strId(r.expression), bits(r.resistance_ohms), bits(r.width_m), bits(r.length_m),
                  bits(r.tc1), bits(r.tc2), r.tc1Given, r.tc2Given, bits(r.m), bits(r.ac_ohms), bits(r.inst.temp), bits(r.inst.dtemp),
                  bits(r.tcFactor)};
            out.push_back({color(k), {r.nodeTerminal1, r.nodeTerminal2}});
        }
        for (const auto& c : b.capacitors) {
            Key k{2, 'C', strId(c.model_name), strId(c.expression), bits(c.capacitance_farads), bits(c.width_m), bits(c.length_m),
                  bits(c.m), bits(c.scale), c.icGiven, bits(c.ic_V), c.tc1Given, c.tc2Given, bits(c.tc1), bits(c.tc2),
                  bits(c.inst.temp), bits(c.inst.dtemp)};
            out.push_back({color(k), {c.nodePlate1, c.nodePlate2}});
        }
        for (const auto& l : b.inductors) {
            Key k{2, 'L', strId(l.model_name), strId(l.expression), bits(l.inductance_henries), bits(l.width_m), bits(l.length_m),
                  bits(l.area_m2), l.turns, l.icGiven, bits(l.ic_A)};
            out.push_back({color(k), {l.nodeCoil1, l.nodeCoil2}});
        }
        auto source = [&](char kind, double dc, const std::string& type, double acm, double acp, const double* pulse,
                          const double* sine, const std::vector<double>& pt, const std::vector<double>& pv, double ptd, double pr,
                          const std::vector<double>& wf, const std::string& expr) {
            Key k{2, kind, bits(dc), strId(type), bits(acm), bits(acp), strId(expr), bits(ptd), bits(pr)};
            for (int i = 0; i < 7; ++i) k.push_back(bits(pulse[i]));
            for (int i = 0; i < 6; ++i) k.push_back(bits(sine[i]));
            add(k, pt); add(k, pv); add(k, wf);
            return color(k);
        };
        for (const auto& v : b.voltageSources) {
            const double pulse[7] = {v.pulse_v1, v.pulse_v2, v.pulse_td, v.pulse_tr, v.pulse_tf, v.pulse_pw, v.pulse_per};
            const double sine[6] = {v.sine_vo, v.sine_va, v.sine_freq, v.sine_td, v.sine_theta, v.sine_phase};
            out.push_back({source('V', v.voltage_V, v.type, v.ac_mag, v.ac_phase, pulse, sine, v.pwl_t, v.pwl_v, v.pwl_td, v.pwl_r,
                                  v.wf, v.expression), {v.nodePositive, v.nodeNegative}});
        }
        for (const auto& v : b.currentSources) {
            const double pulse[7] = {v.pulse_v1, v.pulse_v2, v.pulse_td, v.pulse_tr, v.pulse_tf, v.pulse_pw, v.pulse_per};
            const double sine[6] = {v.sine_vo, v.sine_va, v.sine_freq, v.sine_td, v.sine_theta, v.sine_phase};
            out.push_back({source('I', v.current_A, v.type, v.ac_mag, v.ac_phase, pulse, sine, v.pwl_t, v.pwl_v, v.pwl_td, v.pwl_r,
                                  v.wf, v.expression), {v.nodePositive, v.nodeNegative}});
        }
        // the rest by identity (the same element of the same definition)
        for (size_t i = 0; i < b.diodes.size(); ++i) byIndex('D', i, {b.diodes[i].anode, b.diodes[i].cathode, b.diodes[i].thermal});
        for (size_t i = 0; i < b.mosfets.size(); ++i) {
            const auto& m = b.mosfets[i];
            std::vector<int> t{m.drain, m.gate, m.source, m.body};
            t.insert(t.end(), m.soiNodes.begin(), m.soiNodes.end());
            byIndex('M', i, t);
        }
        for (size_t i = 0; i < b.bjts.size(); ++i) {
            const auto& q = b.bjts[i];
            byIndex('Q', i, {q.nodeCollector, q.base, q.emitter, q.substrate, q.thermal});
        }
        for (size_t i = 0; i < b.osdiDevices.size(); ++i) byIndex('N', i, b.osdiDevices[i].nodes);
        for (size_t i = 0; i < b.jfets.size(); ++i) byIndex('J', i, {b.jfets[i].drain, b.jfets[i].gate, b.jfets[i].source});
        for (size_t i = 0; i < b.switches.size(); ++i) {
            const auto& s = b.switches[i];
            byIndex('S', i, {s.nodePos, s.nodeNeg, s.ctrlPos, s.ctrlNeg});
        }
        for (size_t i = 0; i < b.voltageControlledVoltageSources.size(); ++i) {
            const auto& e = b.voltageControlledVoltageSources[i];
            byIndex('E', i, {e.outPos, e.outNeg, e.ctrlPos, e.ctrlNeg});
        }
        for (size_t i = 0; i < b.voltageControlledCurrentSources.size(); ++i) {
            const auto& g = b.voltageControlledCurrentSources[i];
            byIndex('G', i, {g.outPos, g.outNeg, g.ctrlPos, g.ctrlNeg});
        }
        for (size_t i = 0; i < b.transmissionLines.size(); ++i) {
            const auto& t = b.transmissionLines[i];
            byIndex('T', i, {t.nodePort1Pos, t.nodePort1Neg, t.nodePort2Pos, t.nodePort2Neg});
        }
        for (size_t i = 0; i < b.lossyLines.size(); ++i) {
            const auto& o = b.lossyLines[i];
            byIndex('O', i, {o.p1, o.n1, o.p2, o.n2});
        }
        for (size_t i = 0; i < b.parasiticResistors.size(); ++i) byIndex('r', i, {b.parasiticResistors[i].node1, b.parasiticResistors[i].node2});
        for (size_t i = 0; i < b.parasiticCapacitors.size(); ++i) byIndex('c', i, {b.parasiticCapacitors[i].node1, b.parasiticCapacitors[i].node2});
        return out;
    }

    std::vector<Hierarchy::Seed> topSeeds() const {
        std::vector<Hierarchy::Seed> s;
        for (const auto& [n, v] : nl_.nodeICs) s.push_back({n, v, false});
        for (const auto& [n, v] : nl_.nodeSets) s.push_back({n, v, true});
        return s;
    }

    static std::vector<int> pinsOf(const HierInstance& x, const std::vector<int>& net) {
        std::vector<int> c;
        for (int n : x.ports) c.push_back(n >= 0 && n < (int)net.size() ? net[n] : -1);
        return c;
    }

    // The context of definition def with pin colors pins, refined once
    int context(const std::string& def, const std::vector<int>& pins) {
        const auto key = std::make_pair(def, pins);
        if (auto it = contexts_.find(key); it != contexts_.end()) return it->second;
        const auto bd = nl_.blockDefinitions.find(def);
        if (bd == nl_.blockDefinitions.end()) throw std::runtime_error("unknown subcircuit '" + def + "'");
        static const std::vector<HierInstance> none;
        static const std::vector<Hierarchy::Seed> noSeeds;
        const auto kids = nl_.hier.children.find(def);
        const auto seeds = nl_.hier.seeds.find(def);
        Context c;
        c.def = def;
        c.pins = pins;
        refine(bd->second, bd->second.numInternalNodes, pins, seeds == nl_.hier.seeds.end() ? noSeeds : seeds->second,
               kids == nl_.hier.children.end() ? none : kids->second, c);
        const int id = (int)p_->ctx.size();
        contexts_[key] = id;
        p_->ctx.push_back(c);
        for (size_t k = 0; k < c.child.size(); ++k) {
            const int sub = context((kids->second)[k].def, pinsOf((kids->second)[k], p_->ctx[id].net));
            p_->ctx[id].child[k] = sub;
        }
        return id;
    }

    // Color refinement of one block's nodes 0..N: pins 1..P start at the context's colors, the rest internal
    void refine(const TensorBlock& b, int N, const std::vector<int>& pins, const std::vector<Hierarchy::Seed>& seeds,
                const std::vector<HierInstance>& kids, Context& c) {
        const auto els = elements(b, c.def);
        std::vector<int> col(N + 1);
        col[0] = color({3, 0});
        for (int n = 1; n <= N; ++n) col[n] = n <= (int)pins.size() ? pins[n - 1] : color({3, 1});
        for (const auto& s : seeds)
            if (s.node >= 1 && s.node <= N) col[s.node] = color({4, col[s.node], bits(s.value), s.nodeset});
        // incidence: per node, (element or child, terminal)
        std::vector<std::vector<std::pair<int, int>>> inc(N + 1);  // (index: elements first, then children; terminal)
        for (size_t e = 0; e < els.size(); ++e)
            for (size_t t = 0; t < els[e].t.size(); ++t)
                if (els[e].t[t] >= 0 && els[e].t[t] <= N) inc[els[e].t[t]].push_back({(int)e, (int)t});
        for (size_t k = 0; k < kids.size(); ++k)
            for (size_t t = 0; t < kids[k].ports.size(); ++t)
                if (kids[k].ports[t] >= 0 && kids[k].ports[t] <= N) inc[kids[k].ports[t]].push_back({(int)(els.size() + k), (int)t});
        const int kidDef = 5;
        auto distinct = [](std::vector<int> v) { std::sort(v.begin(), v.end()); return (size_t)(std::unique(v.begin(), v.end()) - v.begin()); };
        size_t classes = distinct(col);
        for (;;) {
            std::vector<int> part(els.size() + kids.size());
            for (size_t e = 0; e < els.size(); ++e) {
                Key k{6, els[e].id};
                for (int t : els[e].t) k.push_back(t >= 0 && t <= N ? col[t] : -1);
                part[e] = color(k);
            }
            for (size_t x = 0; x < kids.size(); ++x) {
                Key k{kidDef, strId(kids[x].def)};
                for (int t : kids[x].ports) k.push_back(t >= 0 && t <= N ? col[t] : -1);
                part[els.size() + x] = color(k);
            }
            std::vector<int> next(N + 1);
            next[0] = col[0];
            for (int n = 1; n <= N; ++n) {
                std::vector<std::pair<int, int>> m;
                for (const auto& [i, t] : inc[n]) m.push_back({part[i], t});
                std::sort(m.begin(), m.end());
                Key k{7, col[n]};
                for (const auto& [pc, t] : m) { k.push_back(pc); k.push_back(t); }
                next[n] = color(k);
            }
            const size_t nc = distinct(next);
            col.swap(next);
            if (nc == classes) break;  // refinement only splits: no new class, fixed point
            classes = nc;
        }
        c.net = col;
        c.child.assign(kids.size(), -1);
    }
};

inline Partition partition(const TensorNetlist& nl) { return Refiner(nl).run(); }

}  // namespace hierpart
