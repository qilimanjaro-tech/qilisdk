// Copyright 2026 Qilimanjaro Quantum Tech
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "stabilizer_state.h"

std::ostream& operator<<(std::ostream& os, const StabilizerStateSum& sss) {
    os << "StabilizerStateSum with " << sss.get_states().size() << " terms:\n";
    for (size_t i = 0; i < sss.get_states().size(); ++i) {
        const int n = sss.get_states()[i].get_nqubits();
        auto truncate = [n](const std::bitset<MAX_ROWS_STABILIZER>& bs) {
            auto s = bs.to_string().substr(MAX_ROWS_STABILIZER - n);
            std::reverse(s.begin(), s.end());
            return s;
        };
        os << "Coefficient: " << sss.get_coefficients()[i] << "\n";
        os << "X stabilizers:\n";
        for (int j = 0; j < n; ++j) {
            os << "For qubit " << j << ": " << truncate(sss.get_states()[i].get_x_bits()[j]) << "\n";
        }
        os << "Z stabilizers:\n";
        for (int j = 0; j < n; ++j) {
            os << "For qubit " << j << ": " << truncate(sss.get_states()[i].get_z_bits()[j]) << "\n";
        }
        os << "Phases: " << truncate(sss.get_states()[i].get_phases()) << "\n";
    }
    return os;
}

std::map<std::string, int> StabilizerStateSum::sample(int nshots) const {
    // TODO(luke): implement sampling from a StabilizerStateSum
    std::string samples_str = "";
    for (size_t i = 0; i < nqubits; ++i) {
        samples_str += "0";
    }
    return {{samples_str, nshots}};
}

void StabilizerState::apply_gate(const Gate& gate) {
    const std::string name = gate.get_name();
    const auto& ctrl = gate.get_control_qubits();
    const auto& tgt = gate.get_target_qubits();

    if (name == "SWAP") {
        int i = tgt[0], j = tgt[1];
        std::swap(x_bits[i], x_bits[j]);
        std::swap(z_bits[i], z_bits[j]);
    } else if (ctrl.empty()) {
        int i = tgt[0];
        if (name == "H") {
            phases ^= (x_bits[i] & z_bits[i]);
            std::swap(x_bits[i], z_bits[i]);
        } else if (name == "S") {
            phases ^= (x_bits[i] & z_bits[i]);
            z_bits[i] ^= x_bits[i];
        } else if (name == "Sdg") {
            phases ^= (x_bits[i] & ~z_bits[i]);
            z_bits[i] ^= x_bits[i];
        } else if (name == "X") {
            phases ^= z_bits[i];
        } else if (name == "Y") {
            phases ^= (x_bits[i] ^ z_bits[i]);
        } else if (name == "Z") {
            phases ^= x_bits[i];
        }
    } else {
        // Two-qubit controlled gates: CNOT is stored as X with ctrl, CZ as Z with ctrl
        int i = ctrl[0], j = tgt[0];
        if (name == "X") {
            // CNOT: ctrl=i, tgt=j — phase computed before modifying x[j], z[i]
            phases ^= (x_bits[i] & z_bits[j] & ~(x_bits[j] ^ z_bits[i]));
            x_bits[j] ^= x_bits[i];
            z_bits[i] ^= z_bits[j];
        } else if (name == "Z") {
            // CZ: qubits i, j — phase computed before modifying z[i], z[j]
            phases ^= (x_bits[i] & x_bits[j] & ~(z_bits[i] ^ z_bits[j]));
            z_bits[j] ^= x_bits[i];
            z_bits[i] ^= x_bits[j];
        }
    }
}

void StabilizerStateSum::apply_gate(const Gate& gate) {
    const std::string name = gate.get_name();
    if (name == "H" || name == "S" || name == "Sdg" || name == "X" || name == "Y" || name == "Z" || name == "SWAP") {
        for (auto& state : states) {
            state.apply_gate(gate);
        }
    } else {
        throw std::runtime_error("Gate " + name + " not supported for StabilizerStateSum yet");
    }
}