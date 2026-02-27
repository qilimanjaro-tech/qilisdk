// Copyright 2025 Qilimanjaro Quantum Tech
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

#include <vector>
#include <map>
#include <set>
#include <complex>
#include "../libs/eigen.h"
#include "../digital/gate.h"
#include "affine_stabilizer.h"

class MatrixFreeOperator {
    private:
        std::string name;
        int target_qubit;
        int control_qubit;
        DenseMatrix base_matrix;
    public:
        MatrixFreeOperator(const Gate& gate);
        void apply(AffineStabilizerState& output_state) const;
        void apply(DenseMatrix& output_state, bool as_density_matrix) const;
        AffineStabilizerState operator*(const AffineStabilizerState& input_state) const;
        friend std::ostream& operator<<(std::ostream& os, const MatrixFreeOperator& mfo);
};