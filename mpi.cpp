#include <iostream>
#include <vector>
#include <omp.h>
#include <mpi.h>
#include <chrono>

using namespace std;

vector<double> generate_vector(const int size, const double first, const double middle, const double last) {
    if (size < 3) {
        cerr << "Size should be at least 3." << std::endl;
        return vector<double>();
    }

    std::vector<double> result(size, 0.0);

    result[0] = first;
    result[size / 2] = middle;
    result[size - 1] = last;

    return result;
}

vector<double> smooth_vector_without_parallel(const vector<double>& input, const double number, const int m) {
    if (input.empty())
    {
        return input;
    }
    const int n = input.size();
    std::vector<double> output = input;
    for (int i = 0; i < m; ++i) {
        std::vector<double> temp = output;
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
    return output;

}


std::vector<double> smooth_vector_with_mpi(std::vector<double>& input, const double number, const int m, int& size, int& rank) {
    if (!input.empty()) {
        const int n = input.size();
        vector<double> output = input;
        for (int i = 0; i < m; i++)
        {
            if (rank > 0) {
                MPI_Send(&input[0], 1, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD);
            }
            if (rank < size - 1)
            {
                MPI_Send(&input[n - 1], 1, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD);
            }
            for (int j = 0; j < n; j++)
            {
                if (input[j] != number) {
                    double prev = 0;
                    double next = 0;
                    if (j == 0 && rank != 0)
                    {
                        MPI_Recv(&prev, 1, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                    if (j == n - 1 && rank != size - 1)
                    {
                        MPI_Recv(&next, 1, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                    if (j != 0)
                    {
                        prev = input[j - 1];
                    }
                    if (j != n - 1)
                    {
                        next = input[j + 1];
                    }
                    output[j] = (prev + next) / 2;
                }
            }
            std::swap(output, input);
        }
    }
    return input;
}

int main(int argc, char** argv)
{
    const int m = 1000;
    const int N = 100000;

    const double number = 1.0;
    std::vector<double> input = generate_vector(N, number, number, number);

    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int chunk_size = N / size;
    std::vector<int> displs(size, 0);
    std::vector<int> sizes(size, chunk_size);

    int tempDisp = 0;
    for (int i = 0; i < size; i++) {
        displs[i] = tempDisp;
        tempDisp += chunk_size;
    }
    if (N % size == 0)
    {
        sizes[size - 1] = chunk_size;
    }
    else
    {
        sizes[size - 1] = chunk_size + N % size;
    }

    std::vector<double> result(sizes[rank]);

    if (rank == 0) {
        auto start_time = std::chrono::high_resolution_clock::now();
        MPI_Scatterv(&input[0], &sizes[0], &displs[0], MPI_DOUBLE, &result[0], sizes[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        smooth_vector_with_mpi(result, number, m, size, rank);
        MPI_Gatherv(&result[0], sizes[rank], MPI_DOUBLE, &input[0], &sizes[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "Execution time: " << static_cast<double>(duration.count()) / 1'000'000.0 << " seconds" << std::endl;
    }

    else {
        MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, &result[0], sizes[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        smooth_vector_with_mpi(result, number, m, size, rank);
        MPI_Gatherv(&result[0], sizes[rank], MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}
