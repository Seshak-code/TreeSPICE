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
        const int top = refineCtx("", {});
        // the contexts reached with the final colors (refinement also met intermediate ones), parents first
        Partition p;
        std::map<int, int> id;
        std::vector<int> order{top};
        id[top] = 0;
        for (size_t k = 0; k < order.size(); ++k)
            for (int c : refs_[order[k]].child)
                if (id.emplace(c, (int)order.size()).second) order.push_back(c);
        for (int r : order) {
            const Ref& f = refs_[r];
            Context c{f.def, f.pins, f.net, {}, f.elem};
            for (int ch : f.child) c.child.push_back(id.at(ch));
            p.ctx.push_back(std::move(c));
        }
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
        // classes: a context's pin is its parent's net (its refined color joins the parent's), any other net its color
        std::map<int, int> parent;
        std::function<int(int)> find = [&](int x) {
            auto it = parent.find(x);
            if (it == parent.end() || it->second == x) return x;
            return it->second = find(it->second);
        };
        auto unite = [&](int a, int b) {
            a = find(a); b = find(b);
            if (a == b) return;
            if (b == ground) std::swap(a, b);  // ground stays the root of its class
            parent[b] = a;
        };
        for (const auto& c : p.ctx) {
            const size_t pins = c.def.empty() ? 0 : block(c).pinNames.size();
            for (size_t n = 1; n <= pins && n < c.net.size(); ++n) unite(c.pins[n - 1], c.net[n]);
        }
        std::map<int, int> rootNode;
        auto nodeOf = [&](int col) {
            const int r = find(col);
            return r == find(ground) ? 0 : rootNode.emplace(r, (int)rootNode.size() + 1).first->second;
        };
        auto classOf = [&](const Context& c, int n) { return c.net[n]; };
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
        for (const auto& c : p.ctx)  // every color of every context, for nodeFor
            for (size_t n = 1; n < c.net.size(); ++n)
                if (const int qn = nodeOf(c.net[n]); qn > 0) q.node[c.net[n]] = qn;
        q.nl.numGlobalNodes = (int)rootNode.size();
        size.resize(rootNode.size(), 0.0);
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
    std::map<Key, int> colors_;                      // descriptor -> color

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

    // A definition refined in the context of its pins' colors (memoized). Pins refine like any net (their refined
    // colors are what a parent's net sees of this child), so a child contributes to its parent's nets by the refined
    // color of the pin there: interchangeable pins (a row's bit lines) stay one class; the quotient joins each pin
    // with its parent's net.
    struct Ref { std::string def; std::vector<int> pins, net, elem, child; };  // child: refs_ indices
    std::vector<Ref> refs_;
    std::map<std::pair<std::string, std::vector<int>>, int> memo_;

    int refineCtx(const std::string& def, const std::vector<int>& pins) {
        const auto key = std::make_pair(def, pins);
        if (auto it = memo_.find(key); it != memo_.end()) return it->second;
        static const std::vector<HierInstance> none;
        static const std::vector<Hierarchy::Seed> noSeeds;
        const bool top = def.empty();
        const auto bd = top ? nl_.blockDefinitions.end() : nl_.blockDefinitions.find(def);
        if (!top && bd == nl_.blockDefinitions.end()) throw std::runtime_error("unknown subcircuit '" + def + "'");
        const TensorBlock& b = top ? nl_.globalBlock : bd->second;
        const int N = top ? nl_.numGlobalNodes : b.numInternalNodes;
        const auto kidsIt = nl_.hier.children.find(def);
        const auto& kids = top ? nl_.hier.top : kidsIt == nl_.hier.children.end() ? none : kidsIt->second;
        const auto seedsIt = nl_.hier.seeds.find(def);
        const std::vector<Hierarchy::Seed> topS = top ? topSeeds() : std::vector<Hierarchy::Seed>{};
        const auto& seeds = top ? topS : seedsIt == nl_.hier.seeds.end() ? noSeeds : seedsIt->second;
        const auto els = elements(b, def);

        std::vector<int> col(N + 1);
        col[0] = color({3, 0});
        for (int n = 1; n <= N; ++n) col[n] = n <= (int)pins.size() ? pins[n - 1] : color({3, 1});
        for (const auto& sd : seeds)
            if (sd.node >= 1 && sd.node <= N) col[sd.node] = color({4, col[sd.node], bits(sd.value), sd.nodeset});
        std::vector<std::vector<std::pair<int, int>>> inc(N + 1);  // per node: (element, terminal) or (els + child, pin)
        for (size_t e = 0; e < els.size(); ++e)
            for (size_t t = 0; t < els[e].t.size(); ++t)
                if (els[e].t[t] >= 0 && els[e].t[t] <= N) inc[els[e].t[t]].push_back({(int)e, (int)t});
        for (size_t k = 0; k < kids.size(); ++k)
            for (size_t t = 0; t < kids[k].ports.size(); ++t)
                if (kids[k].ports[t] >= 0 && kids[k].ports[t] <= N) inc[kids[k].ports[t]].push_back({(int)(els.size() + k), (int)t});
        auto distinct = [](std::vector<int> v) { std::sort(v.begin(), v.end()); return (size_t)(std::unique(v.begin(), v.end()) - v.begin()); };
        std::vector<int> childRef(kids.size());
        size_t classes = distinct(col);
        for (;;) {
            std::vector<int> part(els.size());
            for (size_t e = 0; e < els.size(); ++e) {
                Key k{6, els[e].id};
                for (int t : els[e].t) k.push_back(t >= 0 && t <= N ? col[t] : -1);
                part[e] = color(k);
            }
            for (size_t x = 0; x < kids.size(); ++x) childRef[x] = refineCtx(kids[x].def, pinsOf(kids[x], col));
            std::vector<int> next(N + 1);
            next[0] = col[0];
            for (int n = 1; n <= N; ++n) {
                std::vector<std::pair<int, int>> m;  // (element class, terminal) or (the child's refined pin color, -1)
                for (const auto& [i, t] : inc[n])
                    m.push_back(i < (int)els.size() ? std::make_pair(part[i], t) : std::make_pair(refs_[childRef[i - els.size()]].net[t + 1], -1));
                std::sort(m.begin(), m.end());
                Key k{7, col[n]};
                for (size_t a = 0; a < m.size();) {  // run-length: a bit line meets 1024 equal rows
                    size_t z = a;
                    while (z < m.size() && m[z] == m[a]) ++z;
                    k.push_back(m[a].first); k.push_back(m[a].second); k.push_back((long long)(z - a));
                    a = z;
                }
                next[n] = color(k);
            }
            const size_t nc = distinct(next);
            col.swap(next);
            if (nc == classes) break;  // refinement only splits: no new class, fixed point
            classes = nc;
        }
        // A stable local partition's color ids still carry only as many rounds of history as this context needed, so
        // a net two rounds from a differing pin could share its id with another context's: final colors name the
        // context (its definition and pin colors), so equal colors are one context's one class.
        Key ck{8, strId(def)};
        ck.insert(ck.end(), pins.begin(), pins.end());
        const int self = color(ck);
        for (int n = 1; n <= N; ++n) col[n] = color({9, self, col[n]});
        Ref r;
        r.def = def;
        r.pins = pins;
        r.net = col;
        r.elem.resize(els.size());
        for (size_t e = 0; e < els.size(); ++e) {  // the final classes: element and its terminals' colors
            Key k{6, els[e].id};
            for (int t : els[e].t) k.push_back(t >= 0 && t <= N ? col[t] : -1);
            r.elem[e] = color(k);
        }
        for (size_t x = 0; x < kids.size(); ++x) r.child.push_back(refineCtx(kids[x].def, pinsOf(kids[x], col)));  // final colors
        const int id = (int)refs_.size();
        memo_[key] = id;
        refs_.push_back(std::move(r));
        return id;
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
