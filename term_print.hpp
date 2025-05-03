#ifndef TERM_PRINT_HPP
#define TERM_PRINT_HPP

#include <iostream>
#include <string>

using namespace std;

inline void print_red_str_to_term(string output_string)
{
    cout << "\033[31m" + output_string + "\033[0m" << endl;
}

#endif