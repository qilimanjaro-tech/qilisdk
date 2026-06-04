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
        os << "Coefficient: " << sss.get_coefficients()[i] << "\n";
        os << "X stabilizers:\n";
        for (const auto& x : sss.get_states()[i].get_x_stabilizers()) {
            os << x << "\n";
        }
        os << "Z stabilizers:\n";
        for (const auto& z : sss.get_states()[i].get_z_stabilizers()) {
            os << z << "\n";
        }
        os << "Phases: " << sss.get_states()[i].get_phases() << "\n";
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