#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

std::vector<double> generate_vector(const int size, const double first, const double middle, const double last) {
    if (size < 3) {
        std::cerr << "Size should be at least 3." << std::endl;
        return std::vector<double>();
    }

    std::vector<double> result(size, 0.0);

    result[0] = first;
    result[size / 2] = middle;
    result[size - 1] = last;

    return result;
}

std::vector<double> smooth_vector_with_parallel(const std::vector<double>& input, const double number, const int m, const int number_of_threads) {
    if (input.empty())
    {
        return input;
    }
    const int n = input.size();
    std::vector<double> output = input;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < m; ++i) {
        std::vector<double> temp = output;

        omp_set_num_threads(number_of_threads);
        #pragma omp parallel for 
        for (int j = 1; j < n - 1; ++j) {
            if (output[j] != number) {
                temp[j] = (output[j - 1] + output[j + 1]) / 2.0;
            }
            else
            {
                temp[j] = output[j];
            }
        }
        output = temp;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Execution time: " << static_cast<double>(duration.count()) / 1'000'000.0 << " seconds" << std::endl;
    return output;

}

std::vector<double> smooth_vector_without_parallel(const std::vector<double>& input, const int m) {
    if (input.empty())
    {
        return input;
    }
    const int n = input.size();
    std::vector<double> output = input;
    for (int i = 0; i < m; ++i) {
        std::vector<double> temp = output;
        for (int j = 1; j < n - 1; ++j) {
            if (output[j] == 0.0) {
                temp[j] = (output[j - 1] + output[j + 1]) / 2.0;
            }
            else
            {
                temp[j] = output[j];
            }
        }
        output = temp;
    }
    return output;

}

int main()
{
    const int m = 1000;
    const int N = 2000000;
    const int n = 4;
    const double number = 1.0;
    std::vector<double> input = generate_vector(N, number, number, number);
    std::vector<double> newVector = smooth_vector_with_parallel(input, number, m, n);

    return 0;
}
