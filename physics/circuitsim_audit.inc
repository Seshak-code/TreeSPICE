  std::string auditPhysicalResiduals(const TensorNetlist &netlist, const std::vector<double> &v_candidate) {
    std::string report;
    auto getV = [&](int nodeIdx) {
      if (nodeIdx <= 0 || nodeIdx > (int)v_candidate.size()) return 0.0;
      return v_candidate[nodeIdx - 1];
    };

    // The residual of every element (calculatePhysicalResiduals: the one KCL audit); the warnings below name the
    // large currents behind it
    const int n = netlist.numGlobalNodes;
    std::vector<double> node_residuals;
    calculatePhysicalResiduals(netlist, v_candidate, 0.0, 0.0, &node_residuals);
    node_residuals.resize(n, 0.0);
    for (size_t i = 0; i < netlist.globalBlock.voltageSources.size(); ++i) {
        const int b = n + (int)i;  // global V sources are the first branch rows
        const double i_src = b < (int)v_candidate.size() ? v_candidate[b] : 0.0;
        if (std::abs(i_src) > 100.0) {
            const auto &v = netlist.globalBlock.voltageSources[i];
            report += "WARN: VSource (Nodes " + std::to_string(v.nodePositive) + "," + std::to_string(v.nodeNegative) + ") I=" + std::to_string(i_src) + " A\n";
        }
    }
    forEachJunction(netlist, [&](const Junction &j) {  // terminal currents > 10 A
        std::array<double, kJT> x{};
        for (int k = 0; k < j.terms; ++k) x[k] = getV(j.n[k]);
        const auto I = junctionCurrents(j, x, 0.0);
        const std::string &name = j.d ? j.d->instanceName : j.f ? j.f->instanceName : j.q->instanceName;
        for (int k = 0; k < j.terms; ++k)
            if (std::abs(I[k]) > 10.0)
                report += "WARN: " + name + " terminal " + std::to_string(k) + " I=" + std::to_string(I[k]) + " A\n";
    });

    // Nodes whose KCL is far off
    for (int i=0; i<n; ++i) {
        if (std::abs(node_residuals[i]) > 10.0) {
            report += "FAIL: Node " + std::to_string(i+1) + " KCL Resid=" + std::to_string(node_residuals[i]) + " A. V=" + std::to_string(getV(i+1)) + "\n";
        }
    }

    // Capture Partial Voltages
    if (report.length() > 0) {
        report += "Voltages: ";
        for(size_t i=0; i<v_candidate.size(); ++i) {
            if (i >= 10) { report += "..."; break; } 
            report += std::to_string(v_candidate[i]) + ", ";
        }
        report += "\n";
    }

    return report;
  }
