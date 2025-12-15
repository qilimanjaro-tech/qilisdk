
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <complex>
#include <sstream>
#include <tuple>
#include <map>
#include <random>
#include <stdexcept>
#include <algorithm>

enum class GateType {
    H,
    X,
    Y,
    Z,
    RX,
    RY,
    RZ,
    U1,
    U2,
    U3,
    M,
    CNOT,
    UNKNOWN
};

const double atol_ = 1e-12;

class SparseMatrix {
    /*
    A simple sparse matrix representation using compressed sparse row (CSR) format.
    */
private:
    int nrows_ = 0;
    int ncols_ = 0;
    std::vector<int> rows_;
    std::vector<int> cols_;
    std::vector<std::complex<double>> values_;
public:
    SparseMatrix() {}
    SparseMatrix(int nrows, int ncols) : nrows_(nrows), ncols_(ncols) {
        /*
        Construct an empty sparse matrix of given dimensions.
        Args:
            nrows (int): Number of rows.
            ncols (int): Number of columns.
        */
        rows_.resize(nrows_ + 1, 0);
    }
    SparseMatrix(std::vector<std::vector<std::complex<double>>> dense) {
        /*
        Construct a sparse matrix from a dense 2D vector.
        Args:
            dense (std::vector<std::vector<std::complex<double>>>): The dense matrix.
        */
        nrows_ = dense.size();
        ncols_ = dense[0].size();
        rows_.resize(nrows_ + 1, 0);
        for (int r = 0; r < nrows_; ++r) {
            for (int c = 0; c < ncols_; ++c) {
                if (std::abs(dense[r][c]) > atol_) {
                    cols_.push_back(c);
                    values_.push_back(dense[r][c]);
                    rows_[r + 1]++;
                }
            }
        }
        // Convert counts to offsets
        for (int r = 1; r <= nrows_; ++r) {
            rows_[r] += rows_[r - 1];
        }
    }

    SparseMatrix(std::vector<std::tuple<int, int, std::complex<double>>> entries, int nrows, int ncols) {
        /*
        Construct a sparse matrix from a list of (row, col, value) tuples.
        The entries must be sorted by (row, col).

        Args:
            entries (std::vector<std::tuple<int, int, std::complex<double>>>): The non-zero entries.
            nrows (int): Number of rows.
            ncols (int): Number of columns.

        Throws:
            std::runtime_error: If entries are not sorted.
        */
        
        // Ensure entries are sorted
        bool sorted = true;
        for (size_t i = 1; i < entries.size(); ++i) {
            if (std::get<0>(entries[i]) < std::get<0>(entries[i - 1]) ||
                (std::get<0>(entries[i]) == std::get<0>(entries[i - 1]) &&
                 std::get<1>(entries[i]) < std::get<1>(entries[i - 1]))) {
                sorted = false;
                break;
            }
        }
        if (!sorted) {
            throw std::runtime_error("Entries must be sorted by (row, col)");
        }

        // Set the internal sizes
        nrows_ = nrows;
        ncols_ = ncols;

        // Size of the vectors
        rows_.resize(nrows_ + 2, 0);
        cols_.reserve(entries.size());
        values_.reserve(entries.size());

        // Fill CSR structure
        int currentRow = 0;
        int nnzCount = 0;
        for (auto &e : entries) {

            // Fill missing row offsets up to row r
            while (currentRow < std::get<0>(e)) {
                rows_[currentRow + 1] = nnzCount;
                currentRow++;
            }

            // Add value
            cols_.push_back(std::get<1>(e));
            values_.push_back(std::get<2>(e));
            nnzCount++;

        }

        // Finish remaining rows
        while (currentRow <= nrows_) {
            rows_[currentRow + 1] = nnzCount;
            currentRow++;
        }
    }

    // Get the vector of tuples of (row, col, value)
    std::vector<std::tuple<int, int, std::complex<double>>> to_tuples() const {
        /*
        Convert the sparse matrix to a list of (row, col, value) tuples.
        Returns:
            std::vector<std::tuple<int, int, std::complex<double>>>: The list of non-zero entries.
        */
        std::vector<std::tuple<int, int, std::complex<double>>> entries;
        for (int r = 0; r < nrows_; ++r) {
            for (int idx = rows_[r]; idx < rows_[r + 1]; ++idx) {
                entries.emplace_back(r, cols_[idx], values_[idx]);
            }
        }
        return entries;
    }

    // Iterators
    auto begin() const {
        return values_.begin();
    }
    auto end() const {
        return values_.end();
    }
    auto size() const {
        return values_.size();
    }

    int get_width() const {
        /*
        Get the number of columns in the matrix.
        Returns:
            int: The number of columns.
        */
        return ncols_;
    }

    std::complex<double> get(int row, int col) const {
        /*
        Get the value at (row, col). Returns 0 if not present.
        Args:
            row (int): The row index.
            col (int): The column index.
        Returns:
            std::complex<double>: The value at (row, col) or 0 if not present.
        */
        for (int idx = rows_[row]; idx < rows_[row + 1]; ++idx) {
            if (cols_[idx] == col) {
                return values_[idx];
            }
        }
        return 0.0;
    }

    void insert(int row, int col, std::complex<double> value) {
        /*
        Insert a non-zero value into the sparse matrix, maintaining sorted order.
        Args:
            row (int): The row index.
            col (int): The column index.
            value (std::complex<double>): The value to insert.
        */
        
        // Find the position to insert
        int start = rows_[row];
        int end = rows_[row + 1];
        int insertPos = start;
        while (insertPos < end && cols_[insertPos] < col) {
            insertPos++;
        }
        cols_.insert(cols_.begin() + insertPos, col);
        values_.insert(values_.begin() + insertPos, value);

        // Update row offsets
        for (size_t r = row + 1; r < rows_.size(); ++r) {
            rows_[r]++;
        }
        rows_[row + 1]++;
        
    }

    std::string get_dims() const {
        /*
        Get the dimensions of the matrix as a string.
        Returns:
            std::string: The dimensions in the format "nrows x ncols".
        */
        return std::to_string(nrows_) + "x" + std::to_string(ncols_);
    }

    SparseMatrix operator*(const SparseMatrix& other) const {
        /*
        Multiply two sparse matrices.
        Args:
            other (SparseMatrix): The other matrix to multiply.
        Returns:
            SparseMatrix: The result of the multiplication.
        Raises:
            std::runtime_error: If matrix dimensions do not match for multiplication.
        */
        // Check that they match
        if (ncols_ != other.nrows_) {
            throw std::runtime_error("Matrix dimensions do not match for multiplication: " +
                                        get_dims() + " * " + other.get_dims());
        }
        
        // Perform multiplication
        std::vector<std::tuple<int, int, std::complex<double>>> entries;
        for (size_t r = 0; r < rows_.size() - 1; ++r) {
            for (int idxA = rows_[r]; idxA < rows_[r + 1]; ++idxA) {
                int colA = cols_[idxA];
                std::complex<double> valA = values_[idxA];
                for (int idxB = other.rows_[colA]; idxB < other.rows_[colA + 1]; ++idxB) {
                    int colB = other.cols_[idxB];
                    std::complex<double> valB = other.values_[idxB];
                    entries.emplace_back(r, colB, valA * valB);
                }
            }
        }
        SparseMatrix result(entries, nrows_, other.ncols_);
        return result;
    }

};

// Identity matrix constant
const SparseMatrix I = SparseMatrix({{1, 0}, {0, 1}});

class Gate {
    /*
    A quantum gate with type, control qubits, target qubits, and parameters.
    */
private:
    GateType type;
    std::vector<int> control_qubits;
    std::vector<int> target_qubits;
    std::vector<double> parameters;

    std::vector<std::tuple<int, int, std::complex<double>>> tensor_product(
        const std::vector<std::tuple<int, int, std::complex<double>>>& A,
        const std::vector<std::tuple<int, int, std::complex<double>>>& B,
        int A_width,
        int B_width
    ) const {
        /*
        Compute the tensor product of two sets of (row, col, value) tuples.
        Args:
            A (std::vector<std::tuple<int, int, std::complex<double>>>): First matrix entries.
            B (std::vector<std::tuple<int, int, std::complex<double>>>): Second matrix entries.
            A_width (int): Width of the first matrix (assumed square).
            B_width (int): Width of the second matrix (assumed square).
        Returns:
            std::vector<std::tuple<int, int, std::complex<double>>>: The tensor product entries.
        */
        std::vector<std::tuple<int, int, std::complex<double>>> result;
        int row = 0;
        int col = 0;
        std::complex<double> val = 0.0;
        for (const auto& a : A) {
            for (const auto& b : B) {
                row = std::get<0>(a) * B_width + std::get<0>(b);
                col = std::get<1>(a) * B_width + std::get<1>(b);
                val = std::get<2>(a) * std::get<2>(b);
                result.emplace_back(row, col, val);
            }
        }
        return result;
    }

    int permute_bits(int index, const std::vector<int>& perm) const {
        /*
        Permute the bits of an index according to a given permutation.
        This works by taking each bit from the original index and placing it in the new position
        specified by the permutation vector.
        Also note, the qubit versus bit ordering is reversed (i.e. {1, 0, 2} swaps the 
        first bits (0 and 1), not the last/least-significant).
        Args:
            index (int): The original index.
            perm (std::vector<int>): The permutation vector.
        Returns:
            int: The permuted index.
        */
        int n = perm.size();
        int out = 0;
        for (int old_q = 0; old_q < n; old_q++) {
            int new_q = perm[old_q];
            int old_bit = n - 1 - old_q;
            int new_bit = n - 1 - new_q;
            int bit = (index >> old_bit) & 1;
            out |= (bit << new_bit);
        }
        return out;
    }

    SparseMatrix base_to_full(const SparseMatrix& base_gate,
                              int num_qubits,
                              const std::vector<int>& control_qubits,
                              const std::vector<int>& target_qubits) const {
        /*
        Expand a base gate matrix to the full matrix on the entire qubit register.
        Args:
            base_gate (SparseMatrix): The base gate matrix.
            num_qubits (int): Total number of qubits in the register.
            control_qubits (std::vector<int>): List of control qubit indices.
            target_qubits (std::vector<int>): List of target qubit indices.
        Returns:
            SparseMatrix: The full matrix representation of the gate.
        */

        // Determine how many tensor products we need before and after
        int min_qubit = num_qubits;
        for (int q : control_qubits) {
            if (q < min_qubit) min_qubit = q;
        }
        for (int q : target_qubits) {
            if (q < min_qubit) min_qubit = q;
        }
        int base_gate_qubits = target_qubits.size() + control_qubits.size();
        int needed_before = min_qubit;
        int needed_after = num_qubits - needed_before - base_gate_qubits;
        std::vector<std::tuple<int, int, std::complex<double>>> out_entries = base_gate.to_tuples();

        std::cout << "Gate before controls:" << std::endl;
        for (const auto& entry : out_entries) {
            std::cout << "(" << std::get<0>(entry) << ", " << std::get<1>(entry) << ") = " << std::get<2>(entry) << std::endl;
        }

        // Make it controlled if needed
        // Reminder that control gates look like:
        // 1 0 0 0
        // 0 1 0 0
        // 0 0 G G
        // 0 0 G G
        for (int cq : control_qubits) {
            int delta = 1 << (target_qubits.size());
            std::cout << "Adding control on qubit " << cq << " with delta " << delta << std::endl;
            std::vector<std::tuple<int, int, std::complex<double>>> new_entries;
            for (const auto& entry : out_entries) {
                int row = std::get<0>(entry);
                int col = std::get<1>(entry);
                std::complex<double> val = std::get<2>(entry);
                new_entries.emplace_back(row + delta, col + delta, val);
            }
            for (int i = 0; i < delta; ++i) {
                new_entries.emplace_back(i, i, 1.0);
            }
            out_entries = new_entries;
        }

        std::cout << "Gate before tensor:" << std::endl;
        for (const auto& entry : out_entries) {
            std::cout << "(" << std::get<0>(entry) << ", " << std::get<1>(entry) << ") = " << std::get<2>(entry) << std::endl;
        }

        // Perform the tensor products with the identity
        std::vector<std::tuple<int, int, std::complex<double>>> identity_entries = {{0, 0, 1.0}, {1, 1, 1.0}};
        for (int i = 0; i < needed_before; ++i) {
            out_entries = tensor_product(identity_entries, out_entries, 2, base_gate.get_width());
        }
        for (int i = 0; i < needed_after; ++i) {
            out_entries = tensor_product(out_entries, identity_entries, base_gate.get_width(), 2);
        }

        // Determine the permutation to map qubits to their correct positions
        // perm is initially 0 1 2
        // if we have a CNOT on qubits 0 and 2, we actually did it on 0 and 1 internally
        // so we have 0 2 1 and we need to map back
        std::vector<int> perm(num_qubits);
        for (int q = 0; q < num_qubits; ++q) {
            perm[q] = q;
        }
        // for (int i = 0; i < target_qubits.size(); ++i) {
        //     perm[needed_before + i] = target_qubits[i];
        // }
        // perm = {0, 2, 1};

        // debug output TODO
        std::cout << "Base gate non-zero elements: " << base_gate.size() << std::endl;
        std::cout << "Base gate width: " << base_gate.get_width() << std::endl;
        std::cout << "Num qubits before: " << needed_before << ", after: " << needed_after << std::endl;
        std::cout << "Target qubits: ";
        for (int q : target_qubits) {
            std::cout << q << " ";
        }
        std::cout << std::endl;
        std::cout << "Control qubits: ";
        for (int q : control_qubits) {
            std::cout << q << " ";
        }
        std::cout << std::endl;
        std::cout << "Getting gate full matrix for gate type " << static_cast<int>(type) << std::endl;
        std::cout << "Permutation: ";
        for (int q = 0; q < num_qubits; ++q) {
            std::cout << perm[q] << " ";
        }
        std::cout << std::endl;
        
        // Apply the permutation
        for (int i=0; i<out_entries.size(); ++i) {
            int old_row = std::get<0>(out_entries[i]);
            int old_col = std::get<1>(out_entries[i]);
            int new_row = permute_bits(old_row, perm);
            int new_col = permute_bits(old_col, perm);
            out_entries[i] = std::make_tuple(new_row, new_col, std::get<2>(out_entries[i]));
        }

        // Make sure the entries are sorted
        std::sort(out_entries.begin(), out_entries.end(),
                  [](const std::tuple<int, int, std::complex<double>>& a,
                     const std::tuple<int, int, std::complex<double>>& b) {
                      if (std::get<0>(a) != std::get<0>(b)) {
                          return std::get<0>(a) < std::get<0>(b);
                      } else {
                          return std::get<1>(a) < std::get<1>(b);
                      }
                  });

        std::cout << "Final gate matrix:" << std::endl;
        for (const auto& entry : out_entries) {
            std::cout << "(" << std::get<0>(entry) << ", " << std::get<1>(entry) << ") = " << std::get<2>(entry) << std::endl;
        }

        // Form the matrix and return
        int full_size = 1 << num_qubits;
        return SparseMatrix(out_entries, full_size, full_size);

    }

public:

    // Constructor
    Gate(const GateType& type_,
         const std::vector<int>& controls_,
         const std::vector<int>& targets_,
         const std::vector<double>& params_)
        : type(type_), control_qubits(controls_), target_qubits(targets_), parameters(params_) {}

    SparseMatrix get_base_matrix() const {
        /*
        Get the base matrix representation of the gate.
        Returns:
            SparseMatrix: The base matrix representation of the gate.
        */
        switch (type) {
            case GateType::H:
                return SparseMatrix({{1 / std::sqrt(2), 1 / std::sqrt(2)},
                                     {1 / std::sqrt(2), -1 / std::sqrt(2)}});
            case GateType::X:
                return SparseMatrix({{0, 1},
                                     {1, 0}});
            case GateType::Y:
                return SparseMatrix({{0, std::complex<double>(0, -1)},
                                     {std::complex<double>(0, 1), 0}});
            case GateType::Z:
                return SparseMatrix({{1, 0},
                                     {0, -1}});
            case GateType::CNOT:
                return SparseMatrix({{0, 1},
                                     {1, 0}});
            default:
                throw std::runtime_error("Unsupported gate type");
        }
    }

    size_t size() const {
        /*
        Get the size of the base gate matrix.
        Returns:
            int: The size of the base gate matrix.
        */
        return get_base_matrix().size();
    }   

    SparseMatrix get_full_matrix(int num_qubits) const {
        /*
        Get the full matrix representation of the gate on the entire qubit register.
        Args:
            num_qubits (int): Total number of qubits in the register.
        Returns:
            SparseMatrix: The full matrix representation of the gate.
        */
        SparseMatrix base_gate = get_base_matrix();
        SparseMatrix full_gate = base_to_full(base_gate, num_qubits, control_qubits, target_qubits);
        return full_gate;
    }
};

class QiliSim {
public:
    void execute_sampling(const char* functional_string) {

        // Basic circuit info
        std::istringstream iss(functional_string);
        iss >> nshots_;
        iss >> nqubits_;
        int ngates;
        iss >> ngates;
        
        // Fill the circuit
        circuit_.clear();
        for (int i = 0; i < ngates; ++i) {
            
            // Gate name / type
            std::string gate_name;
            iss >> gate_name;
            GateType gate_type = GateType::UNKNOWN;
            if (gate_name == "H") gate_type = GateType::H;
            else if (gate_name == "X") gate_type = GateType::X;
            else if (gate_name == "Y") gate_type = GateType::Y;
            else if (gate_name == "Z") gate_type = GateType::Z;
            else if (gate_name == "RX") gate_type = GateType::RX;
            else if (gate_name == "RY") gate_type = GateType::RY;
            else if (gate_name == "RZ") gate_type = GateType::RZ;
            else if (gate_name == "U1") gate_type = GateType::U1;
            else if (gate_name == "U2") gate_type = GateType::U2;
            else if (gate_name == "U3") gate_type = GateType::U3;
            else if (gate_name == "M") gate_type = GateType::M;
            else if (gate_name == "CNOT") gate_type = GateType::CNOT;

            // Gate info
            int n_controls, n_targets, n_params;
            iss >> n_controls >> n_targets >> n_params;

            // Controls
            std::vector<int> controls(n_controls);
            for (int j = 0; j < n_controls; ++j) {
                iss >> controls[j];
            }
            
            // Targets
            std::vector<int> targets(n_targets);
            for (int j = 0; j < n_targets; ++j) {
                iss >> targets[j];
            }
            
            // Parameters
            std::vector<double> params(n_params);
            for (int j = 0; j < n_params; ++j) {
                iss >> params[j];
            }
            
            // Add the gate to the circuit
            circuit_.emplace_back(gate_type, controls, targets, params);

        }

        // Start with the zero state
        int mat_size = 1 << nqubits_;
        SparseMatrix state({std::make_tuple(0, 0, 1.0)}, mat_size, 1);

        std::cout << "Executing sampling with " << nshots_ << " shots on " << nqubits_ << " qubits and " << ngates << " gates." << std::endl;

        // Apply each gate in the circuit
        for (const auto& gate : circuit_) {
            SparseMatrix gate_matrix = gate.get_full_matrix(nqubits_);
            state = gate_matrix * state;
        }

        // Get the probabilities
        std::vector<std::tuple<int, int, std::complex<double>>> amplitude_entries = state.to_tuples();
        std::vector<std::tuple<int, double>> prob_entries;
        double total_prob = 0.0;
        for (const auto& entry : amplitude_entries) {
            int row = std::get<0>(entry);
            std::complex<double> amp = std::get<2>(entry);
            double prob = std::norm(amp);
            if (prob > atol_) {
                prob_entries.emplace_back(row, prob);
                total_prob += prob;
            }
        }

        // Make sure probabilities sum to 1
        if (std::abs(total_prob - 1.0) > atol_) {
            throw std::runtime_error("Probabilities do not sum to 1 (sum = " + std::to_string(total_prob) + ")");
        }

        // Sample from the final state
        std::map<int, std::string> binary_strings;
        std::map<std::string, int> counts;
        std::string current_bitstring = "";
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        for (int shot = 0; shot < nshots_; ++shot) {
            double random_value = distribution(generator);
            double cumulative_prob = 0.0;
            for (const auto& entry : prob_entries) {
                int state_index = std::get<0>(entry);
                double prob = std::get<1>(entry);
                cumulative_prob += prob;
                if (random_value <= cumulative_prob) {
                    if (binary_strings.find(state_index) == binary_strings.end()) {
                        std::string bitstring = "";
                        for (int b = nqubits_ - 1; b >= 0; --b) {
                            bitstring += ((state_index >> b) & 1) ? '1' : '0';
                        }
                        binary_strings[state_index] = bitstring;
                    }
                    counts[binary_strings[state_index]]++;
                    break;
                }
            }
        }

        // Convert the shots to a result string
        std::ostringstream oss;
        oss << counts.size();
        oss << " " << nshots_;
        for (const auto& pair : counts) {
            oss << " " << pair.first << " " << pair.second;
        }
        result_ = oss.str();

    }

    void execute_time_evolution(const char* functional_string) {
        result_ = std::string(functional_string) + "_time_evolved";
    }

    size_t get_result_size() const {
        return result_.size() + 1; // include null terminator
    }

    void get_result(char* out) const {
        memcpy(out, result_.c_str(), result_.size() + 1);
    }

private:
    std::string result_;
    int nshots_;
    int nqubits_;
    std::vector<Gate> circuit_;
};


extern "C" {

    QiliSim* qilisim_create() {
        return new QiliSim();
    }

    void qilisim_free(QiliSim* s) {
        if (s == nullptr) {
            return;
        }
        delete s;
    }

    void qilisim_execute_sampling(QiliSim* s, const char* functional_string) {
        s->execute_sampling(functional_string);
    }

    void qilisim_execute_time_evolution(QiliSim* s, const char* functional_string) {
        s->execute_time_evolution(functional_string);
    }

    size_t qilisim_get_return_size(QiliSim* s) {
        return s->get_result_size();
    }

    void qilisim_get_return_buffer(QiliSim* s, char* buffer) {
        s->get_result(buffer);
    }

}
