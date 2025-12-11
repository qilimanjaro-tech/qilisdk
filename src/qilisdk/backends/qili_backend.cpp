
#include <iostream>
#include <string>


extern "C" {

    char* _execute_sampling(const char* functional_string) {
        std::string functional_string_str(functional_string);
        std::cout << "Executing sampling functional: " << functional_string_str << std::endl;
        functional_string_str += "_result";
        return functional_string_str.data();
    }

    char* _execute_time_evolution(char* functional_string) {
        std::cout << "Executing time evolution functional: " << functional_string << std::endl;
        return functional_string;
    }

}
