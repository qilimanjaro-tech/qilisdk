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
#pragma once

#include <complex>
#include <map>
#include <set>
#include <vector>
#include "../digital/gate.h"
#include "../libs/eigen.h"

enum class MatrixFreeApplicationType { Left, Right, LeftAndRight };

class MatrixFreeOperator {
   private:
    std::string name;
    std::vector<int> target_qubits;
    std::vector<int> control_qubits;
    DenseMatrix base_matrix;

   public:
    MatrixFreeOperator(const Gate& gate);
    MatrixFreeOperator(const std::string& name, int target_qubit) : name(name), target_qubits({target_qubit}) {}
    void apply(DenseMatrix& output_state, MatrixFreeApplicationType application_type) const;
    friend std::ostream& operator<<(std::ostream& os, const MatrixFreeOperator& mfo);
    std::vector<int> get_target_qubits() const { return target_qubits; }
    std::vector<int> get_control_qubits() const { return control_qubits; }
    std::string get_name() const { return name; }
    std::string get_id() const;
};
