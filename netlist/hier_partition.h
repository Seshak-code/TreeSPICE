#pragma once
// S3.1 (docs/scale-s3.md): the classes of an unflattened netlist's nets (TensorNetlist::hier) on which its solution
// is constant. Color refinement (one-dimensional Weisfeiler-Leman) of each definition in each context it is
// instantiated in: a context is a definition with the colors of its pins seen from its parent (a pin keeps it: it is
// its parent's net). Inside a context a net's color is refined by the colors of the elements it touches (at which
// terminal) and of the child instances it
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

#include <cmath>
#include <cstring>
#include <type_traits>
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
    std::vector<int> elem;    // each element's class (forEachElement order)
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

// Every element of b with its kind, index and node fields (B: TensorBlock or const TensorBlock): one list of the
// element kinds and their terminals for the partition (reads) and the quotient (rewrites)
template <class B, class F> void forEachElement(B& b, F&& f) {
    using I = std::conditional_t<std::is_const_v<B>, const int, int>;
    auto each = [&](char k, auto& v, auto&& terms) { for (size_t i = 0; i < v.size(); ++i) f(k, i, terms(v[i])); };
    each('R', b.resistors, [](auto& e) { return std::vector<I*>{&e.nodeTerminal1, &e.nodeTerminal2}; });
    each('C', b.capacitors, [](auto& e) { return std::vector<I*>{&e.nodePlate1, &e.nodePlate2}; });
    each('L', b.inductors, [](auto& e) { return std::vector<I*>{&e.nodeCoil1, &e.nodeCoil2}; });
    each('V', b.voltageSources, [](auto& e) { return std::vector<I*>{&e.nodePositive, &e.nodeNegative}; });
    each('I', b.currentSources, [](auto& e) { return std::vector<I*>{&e.nodePositive, &e.nodeNegative}; });
    each('D', b.diodes, [](auto& e) { return std::vector<I*>{&e.anode, &e.cathode, &e.thermal}; });
    each('M', b.mosfets, [](auto& e) {
        std::vector<I*> t{&e.drain, &e.gate, &e.source, &e.body};
        for (auto& n : e.soiNodes) t.push_back(&n);
        return t;
    });
    each('Q', b.bjts, [](auto& e) { return std::vector<I*>{&e.nodeCollector, &e.base, &e.emitter, &e.substrate, &e.thermal}; });
    each('N', b.osdiDevices, [](auto& e) { std::vector<I*> t; for (auto& n : e.nodes) t.push_back(&n); return t; });
    each('J', b.jfets, [](auto& e) { return std::vector<I*>{&e.drain, &e.gate, &e.source}; });
    each('S', b.switches, [](auto& e) { return std::vector<I*>{&e.nodePos, &e.nodeNeg, &e.ctrlPos, &e.ctrlNeg}; });
    each('E', b.voltageControlledVoltageSources, [](auto& e) { return std::vector<I*>{&e.outPos, &e.outNeg, &e.ctrlPos, &e.ctrlNeg}; });
    each('G', b.voltageControlledCurrentSources, [](auto& e) { return std::vector<I*>{&e.outPos, &e.outNeg, &e.ctrlPos, &e.ctrlNeg}; });
    each('T', b.transmissionLines, [](auto& e) { return std::vector<I*>{&e.nodePort1Pos, &e.nodePort1Neg, &e.nodePort2Pos, &e.nodePort2Neg}; });
    each('O', b.lossyLines, [](auto& e) { return std::vector<I*>{&e.p1, &e.n1, &e.p2, &e.n2}; });
    each('r', b.parasiticResistors, [](auto& e) { return std::vector<I*>{&e.node1, &e.node2}; });
    each('c', b.parasiticCapacitors, [](auto& e) { return std::vector<I*>{&e.node1, &e.node2}; });
}

// S3.2: the quotient circuit of a partition: one node per net class, one element per element class standing for all
// of its members (|E|: R / |E|, C, MOS1, OSDI and diode multipliers x |E|, a V source's m), each node's KCL row a
// class sum that the solver divides back to one net's (TensorNetlist::classSize). Other kinds with more than one
// member are refused (not supported unflattened yet).
struct Quotient {
    TensorNetlist nl;              // flat: the global block only
    std::map<int, int> node;       // net color -> quotient node (1-based; ground's color: none)
    std::map<std::string, int> vsource;  // a top-level V source's name -> its class's (global block index)
    std::map<std::string, double> topCount;  // a top-level R / V / I's name (upper case) -> its class's members
    size_t elementClasses = 0;
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

    // The quotient circuit of partition p (run's): see Quotient
    Quotient quotient(const Partition& p) {
        Quotient q;
        const int ground = color({3, 0});
        const size_t C = p.ctx.size();
        auto block = [&](const Context& c) -> const TensorBlock& { return c.def.empty() ? nl_.globalBlock : nl_.blockDefinitions.at(c.def); };
        // instances of each context in the expansion; a context reached from several parents sums them (parents first)
        std::vector<double> mult(C, 0.0);
        std::vector<int> indeg(C, 0), ready{0};
        mult[0] = 1.0;
        for (const auto& c : p.ctx) for (int k : c.child) indeg[k]++;
        while (!ready.empty()) {
            const int c = ready.back();
            ready.pop_back();
            for (int k : p.ctx[c].child) { mult[k] += mult[c]; if (--indeg[k] == 0) ready.push_back(k); }
        }
        auto nodeOf = [&](int col) { return col == ground ? 0 : q.node.emplace(col, (int)q.node.size() + 1).first->second; };
        auto classOf = [&](const Context& c, int n) { return c.net[n]; };  // a pin's color is its parent's net's
        std::vector<double> size;  // nets per quotient node: each context's own nodes (not its pins), times its instances
        for (size_t ci = 0; ci < C; ++ci) {
            const auto& c = p.ctx[ci];
            const size_t pins = c.def.empty() ? 0 : block(c).pinNames.size();
            for (size_t n = 1; n < c.net.size(); ++n) {
                const int node = nodeOf(classOf(c, (int)n));
                if (node == 0 || n <= pins) continue;
                if ((int)size.size() < node) size.resize(node, 0.0);
                size[node - 1] += mult[ci];
            }
        }
        // element classes: a representative and its members
        struct ElemClass { size_t ctx, e; double count; };
        std::map<int, ElemClass> classes;
        std::vector<int> order;
        std::vector<std::vector<El>> els(C);
        for (size_t ci = 0; ci < C; ++ci) {
            els[ci] = elements(block(p.ctx[ci]), p.ctx[ci].def);
            for (size_t e = 0; e < els[ci].size(); ++e) {
                const auto [it, isNew] = classes.emplace(p.ctx[ci].elem[e], ElemClass{ci, e, 0.0});
                if (isNew) order.push_back(p.ctx[ci].elem[e]);
                it->second.count += mult[ci];
            }
        }
        TensorBlock& qb = q.nl.globalBlock;
        std::map<char, std::vector<std::pair<size_t, size_t>>> src;  // per kind, per quotient element: (context, its index)
        std::map<int, int> vIndex;  // V source class -> its quotient index
        for (int col : order) {
            const auto& ec = classes.at(col);
            const auto& b = block(p.ctx[ec.ctx]);
            const El& el = els[ec.ctx][ec.e];
            const double m = ec.count;
            const size_t i = el.index;
            auto single = [&](const char* what) {
                if (m != 1.0) throw std::runtime_error(std::string(what) + " standing for " + std::to_string((long long)m) +
                                                       " identical ones is not supported unflattened yet");
            };
            switch (el.kind) {
            case 'R': { auto x = b.resistors[i]; x.resistance_ohms /= m; if (!std::isnan(x.ac_ohms)) x.ac_ohms /= m; qb.resistors.push_back(x); break; }
            case 'C': { auto x = b.capacitors[i]; x.capacitance_farads *= m; qb.capacitors.push_back(x); break; }
            case 'L': single("an inductor"); qb.inductors.push_back(b.inductors[i]); break;
            case 'V': {
                auto x = b.voltageSources[i];
                x.m = m;
                if (ec.ctx != 0) x.name = p.ctx[ec.ctx].def + "." + x.name;  // a body's: named by its definition
                vIndex[col] = (int)qb.voltageSources.size();
                qb.voltageSources.push_back(x);
                break;
            }
            case 'I': single("a current source"); qb.currentSources.push_back(b.currentSources[i]); break;
            case 'D': {
                auto x = b.diodes[i];
                if (x.heat && m != 1.0) single("a self-heating diode");
                x.area *= m; x.mult *= m; x.rep *= m;
                qb.diodes.push_back(x);
                break;
            }
            case 'M': { auto x = b.mosfets[i]; x.extra.mult *= m; x.rep *= m; qb.mosfets.push_back(x); break; }
            case 'N': {
                auto x = b.osdiDevices[i];
                bool given = false;
                for (auto& [k, v] : x.params) if (k == "m" || k == "M") { v *= m; given = true; }
                if (!given && m != 1.0) x.params.push_back({"m", m});
                qb.osdiDevices.push_back(x);
                break;
            }
            case 'Q': single("a BJT"); qb.bjts.push_back(b.bjts[i]); break;
            case 'J': single("a JFET"); qb.jfets.push_back(b.jfets[i]); break;
            case 'S': single("a switch"); qb.switches.push_back(b.switches[i]); break;
            case 'E': single("an E source"); qb.voltageControlledVoltageSources.push_back(b.voltageControlledVoltageSources[i]); break;
            case 'G': single("a G source"); qb.voltageControlledCurrentSources.push_back(b.voltageControlledCurrentSources[i]); break;
            case 'T': single("a transmission line"); qb.transmissionLines.push_back(b.transmissionLines[i]); break;
            case 'O': single("a lossy line"); qb.lossyLines.push_back(b.lossyLines[i]); break;
            case 'r': single("a parasitic resistor"); qb.parasiticResistors.push_back(b.parasiticResistors[i]); break;
            case 'c': single("a parasitic capacitor"); qb.parasiticCapacitors.push_back(b.parasiticCapacitors[i]); break;
            default: throw std::runtime_error(std::string("element kind ") + el.kind + " is not supported unflattened yet");
            }
            src[el.kind].push_back({ec.ctx, i});
        }
        q.elementClasses = order.size();
        auto up = [](std::string x) { for (auto& ch : x) ch = (char)std::toupper((unsigned char)ch); return x; };
        for (size_t e = 0; e < els[0].size(); ++e) {  // the top level's by name (identical ones share a class)
            const auto& el = els[0][e];
            const double members = classes.at(p.ctx[0].elem[e]).count;
            if (el.kind == 'V') {
                const auto& name = nl_.globalBlock.voltageSources[el.index].name;
                q.vsource[name] = vIndex.at(p.ctx[0].elem[e]);
                q.topCount[up(name)] = members;
            }
            if (el.kind == 'I') q.topCount[up(nl_.globalBlock.currentSources[el.index].name)] = members;
            if (el.kind == 'R') q.topCount[up(nl_.globalBlock.resistors[el.index].name)] = members;
        }
        forEachElement(qb, [&](char kind, size_t i, const std::vector<int*>& t) {  // terminals: local nodes -> classes
            const auto& c = p.ctx[src.at(kind)[i].first];
            for (int* n : t)
                if (*n > 0) *n = nodeOf(classOf(c, *n));
        });
        q.nl.numGlobalNodes = (int)q.node.size();
        size.resize(q.node.size(), 0.0);
        for (double z : size)
            if (!(z >= 1.0)) throw std::runtime_error("quotient: a net class with no nets (internal error)");
        q.nl.classSize = std::move(size);
        // initial conditions: the top level's, and each context's body .ic / .nodeset (one value per class)
        std::map<int, double> ics, sets;
        for (const auto& [n, v] : nl_.nodeICs) if (n > 0) ics[nodeOf(classOf(p.ctx[0], n))] = v;
        for (const auto& [n, v] : nl_.nodeSets) if (n > 0) sets[nodeOf(classOf(p.ctx[0], n))] = v;
        for (const auto& c : p.ctx) {
            const auto sd = nl_.hier.seeds.find(c.def);
            if (c.def.empty() || sd == nl_.hier.seeds.end()) continue;
            for (const auto& s : sd->second) (s.nodeset ? sets : ics)[nodeOf(classOf(c, s.node))] = s.value;
        }
        q.nl.nodeICs.assign(ics.begin(), ics.end());
        q.nl.nodeSets.assign(sets.begin(), sets.end());
        q.nl.environment = nl_.environment;
        q.nl.nodeNames.assign(q.nl.numGlobalNodes + 1, "");
        q.nl.globalState.initialize(qb);
        q.nl.instantiateOsdi(q.nl.environment.ambient_temp_K, q.nl.environment.bypass);
        return q;
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

    struct El { int id; std::vector<int> t; char kind; size_t index; };

    // The block's elements with their identity and ordered terminals; refuses what the refinement does not model
    std::vector<El> elements(const TensorBlock& b, const std::string& def) {
        const std::string where = def.empty() ? "the top level" : def;
        if (!b.behavioural.empty() || !b.currentControlledVoltageSources.empty() || !b.currentControlledCurrentSources.empty() ||
            !b.mutualInductors.empty())
            throw std::runtime_error(where + ": B/E/G expressions, F/H and K are not supported unflattened yet");
        for (const auto& s : b.switches)
            if (!s.ctrlName.empty()) throw std::runtime_error(where + ": W switches are not supported unflattened yet");
        // R, C, L, V, I are equal by value, every field (the build fails when one gains a field: keep this whole);
        // the rest by identity, the same element of the same definition
        static_assert(fields<Resistor>() == 16 && fields<Capacitor>() == 17 && fields<Inductor>() == 12 &&
                          fields<VoltageSource>() == 27 && fields<CurrentSource>() == 26,
                      "an R/C/L/V/I struct gained or lost a field: update its descriptor in hier_partition.h (and the count here)");
        auto source = [&](char kind, double dc, const std::string& type, double acm, double acp, const double* pulse,
                          const double* sine, const std::vector<double>& pt, const std::vector<double>& pv, double ptd, double pr,
                          const std::vector<double>& wf, const std::string& expr, double m) {
            Key k{2, kind, bits(dc), strId(type), bits(acm), bits(acp), strId(expr), bits(ptd), bits(pr), bits(m)};
            for (int i = 0; i < 7; ++i) k.push_back(bits(pulse[i]));
            for (int i = 0; i < 6; ++i) k.push_back(bits(sine[i]));
            add(k, pt); add(k, pv); add(k, wf);
            return color(k);
        };
        const int d = strId(def);
        std::vector<El> out;
        forEachElement(b, [&](char kind, size_t i, const std::vector<const int*>& t) {
            int id;
            if (kind == 'R') {
                const auto& r = b.resistors[i];
                id = color({2, 'R', strId(r.model_name), strId(r.expression), bits(r.resistance_ohms), bits(r.width_m),
                            bits(r.length_m), bits(r.tc1), bits(r.tc2), r.tc1Given, r.tc2Given, bits(r.m), bits(r.ac_ohms),
                            bits(r.inst.temp), bits(r.inst.dtemp), bits(r.tcFactor)});
            } else if (kind == 'C') {
                const auto& c = b.capacitors[i];
                id = color({2, 'C', strId(c.model_name), strId(c.expression), bits(c.capacitance_farads), bits(c.width_m),
                            bits(c.length_m), bits(c.m), bits(c.scale), c.icGiven, bits(c.ic_V), c.tc1Given, c.tc2Given,
                            bits(c.tc1), bits(c.tc2), bits(c.inst.temp), bits(c.inst.dtemp)});
            } else if (kind == 'L') {
                const auto& l = b.inductors[i];
                id = color({2, 'L', strId(l.model_name), strId(l.expression), bits(l.inductance_henries), bits(l.width_m),
                            bits(l.length_m), bits(l.area_m2), l.turns, l.icGiven, bits(l.ic_A)});
            } else if (kind == 'V') {
                const auto& v = b.voltageSources[i];
                const double pulse[7] = {v.pulse_v1, v.pulse_v2, v.pulse_td, v.pulse_tr, v.pulse_tf, v.pulse_pw, v.pulse_per};
                const double sine[6] = {v.sine_vo, v.sine_va, v.sine_freq, v.sine_td, v.sine_theta, v.sine_phase};
                id = source('V', v.voltage_V, v.type, v.ac_mag, v.ac_phase, pulse, sine, v.pwl_t, v.pwl_v, v.pwl_td, v.pwl_r, v.wf,
                            v.expression, v.m);
            } else if (kind == 'I') {
                const auto& v = b.currentSources[i];
                const double pulse[7] = {v.pulse_v1, v.pulse_v2, v.pulse_td, v.pulse_tr, v.pulse_tf, v.pulse_pw, v.pulse_per};
                const double sine[6] = {v.sine_vo, v.sine_va, v.sine_freq, v.sine_td, v.sine_theta, v.sine_phase};
                id = source('I', v.current_A, v.type, v.ac_mag, v.ac_phase, pulse, sine, v.pwl_t, v.pwl_v, v.pwl_td, v.pwl_r, v.wf,
                            v.expression, 1.0);
            } else {
                id = color({1, kind, d, (long long)i});
            }
            std::vector<int> terms;
            for (const int* n : t) terms.push_back(*n);
            out.push_back({id, std::move(terms), kind, i});
        });
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
            for (int n = 1; n <= (int)pins.size() && n <= N; ++n) next[n] = col[n];  // a pin stays its parent's net
            for (int n = (int)pins.size() + 1; n <= N; ++n) {
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
        c.elem.resize(els.size());
        for (size_t e = 0; e < els.size(); ++e) {  // the final classes: element and its terminals' colors
            Key k{6, els[e].id};
            for (int t : els[e].t) k.push_back(t >= 0 && t <= N ? col[t] : -1);
            c.elem[e] = color(k);
        }
    }
};

inline Partition partition(const TensorNetlist& nl) { return Refiner(nl).run(); }
// The quotient node of a netlist name: a top-level node (topIndex: the parse's nodeIndex, upper case) or a
// hierarchical one X1.X2.N; 0 for ground, -1 if unknown
inline int nodeFor(const TensorNetlist& nl, const std::map<std::string, int>& topIndex, const Partition& p, const Quotient& q,
                   const std::string& name) {
    int ctx = 0, local = -1;
    if (name.find('.') == std::string::npos) {
        const auto it = topIndex.find(name);
        if (it == topIndex.end()) return -1;
        local = it->second;
    } else {
        std::vector<const HierInstance*> chain;
        if (!nl.hier.resolve(name, chain, local)) return -1;
        const std::vector<HierInstance>* level = &nl.hier.top;
        for (const auto* x : chain) {
            if (!level) return -1;
            ctx = p.ctx[ctx].child[(size_t)(x - level->data())];
            const auto kids = nl.hier.children.find(x->def);
            level = kids == nl.hier.children.end() ? nullptr : &kids->second;
        }
    }
    if (local <= 0) return local == 0 ? 0 : -1;
    const auto it = q.node.find(p.ctx[ctx].net[local]);
    return it == q.node.end() ? 0 : it->second;  // not a class: ground's color
}

// S3.2: the partition (into *part when given) and its quotient circuit
inline Quotient quotient(const TensorNetlist& nl, Partition* part = nullptr) {
    Refiner r(nl);
    Partition p = r.run();
    Quotient q = r.quotient(p);
    if (part) *part = std::move(p);
    return q;
}

}  // namespace hierpart
