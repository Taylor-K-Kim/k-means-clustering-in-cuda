// Taylor Kim @893438416
// CPSC479 Project #2: Implementing K-Means algorithm using CUDA
//                     with multidimensional datasets.
//
// Date: 12/08/2023
//
// NOTE: THIS IS FOR IRIS DATASET
//
// FOR CLUSTERING, from the original dataset last categorical column 
// removed from original dataset as k-means can only work with numerical values.
// However, at the end of the program, resulting cluster numbers are appended
// to the original dataset to check for clustering correctness.


#include <stdio.h>
#include <math.h>   // INFINITY
#include <time.h>   // for rand()
#include <string.h>
#include <cuda.h>

#define N 150   // number of data points
#define F 4     // number of features in dataset
#define K 3     // number of clusters and centroids
#define T 256   // threads per block, ensures enough block
#define MAX_ITER 50    // instead of checking for convergence, not good


__global__ void assignToCluster(float *d_datapoints, float *d_centroids, int *d_assign) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float min_distance = INFINITY;
        int closest_centroid = -1;

        for (int i = 0; i < K; i++) {
            // calculate euclidean distance
            float distance = 0.0;
            for (int j = 0; j < F; j++) {
                float x_y = d_centroids[i * F + j] - d_datapoints[idx * F + j];
                distance += x_y * x_y;
            }

            distance = sqrt(distance);

            // Update minimum distance
            if (distance < min_distance) {
                min_distance = distance;
                closest_centroid = i;
            }
        }

        // assign datapoint to the closest centroid
        d_assign[idx] = closest_centroid;
    }
}

__global__ void updateCentroids(float *d_datapoints, int *d_assign, float *d_centroids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < K) {
        float sum[F] = {0.0};
        int assigned_datapoints = 0;

        for (int i = 0; i < N; i++) {
            if (d_assign[i] == idx) {
                for (int j = 0; j < F; j++) {
                    sum[j] += d_datapoints[i * F + j];
                }
                assigned_datapoints++;
            }
        }

        // update centroid
        if (assigned_datapoints > 0) {
            for (int k = 0; k < F; k++) {
                d_centroids[idx * F + k] = sum[k] / assigned_datapoints;
            }
        }
    }
}

int main() {
    // host memory for datapoints, centroids, and
    // datapoint assignment to any cluster
    float *h_datapoints = (float*)malloc(N*F*sizeof(float));
    float *h_centroids = (float*)malloc(K*F*sizeof(float));
    int *h_assign = (int*)malloc(N*sizeof(int));

    // device memory for datapoints, centroids, and assignment
    float *d_datapoints, *d_centroids;
    int *d_assign;
    cudaMalloc((void **)&d_datapoints, N * F * sizeof(float));
    cudaMalloc((void **)&d_centroids, K * F * sizeof(float));
    cudaMalloc((void **)&d_assign, N * sizeof(int));

    // dataset file name
    // Load dataset
    FILE* dataset = fopen("iris_four_features.csv", "r");
    // FILE* dataset = fopen("winequality_11_features.csv", "r");
    if (dataset == NULL) {
        fprintf(stderr, "Error opening file.");
        exit(EXIT_FAILURE);
    }
    for (int i=0; i<N; i++) {
        for (int j=0; j<F; j++) {
            if (fscanf(dataset, "%f,", &h_datapoints[i*F+j]) != 1) {
                fprintf(stderr, "Error reading datapoints from file.\n");
                exit(EXIT_FAILURE);
            }
        }
    }
    // close file
    fclose(dataset);

    // initialize centroids randomly 
    int randIndex = 0;
    int i, j;
    srand(time(NULL));

    // allocate memeory for centroids randomly chosen
    int *randCentroids = (int*)malloc(K * sizeof(int));

    for (i = 0; i < K; i++) {
        do {
            randIndex = rand() % N; 
            // Check for duplicate
            for (j = 0; j < i; j++) {
                if (randIndex == randCentroids[j]) {
                    break;
                }
            }
        } while (j < i);
        // Save the centroid
        randCentroids[i] = randIndex;

        // Copy the random centroid to initialize
        for (j = 0; j < F; j++) {
           h_centroids[i * F + j] = h_datapoints[randIndex * F + j];
        }
    }

    free(randCentroids);

    // copy host datapoints and centroids to device memory
    cudaMemcpy(d_datapoints, h_datapoints, N * F * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, K * F * sizeof(float), cudaMemcpyHostToDevice);
    
    // repeat assignment and update until converges
    // checking for convergence not implemented
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        // Assign points to centroids
        assignToCluster<<<(N + T - 1) / T, T>>>(d_datapoints, d_centroids, d_assign);

        // Update centroids based on assigned points
        updateCentroids<<<(K + T - 1) / T, T>>>(d_datapoints, d_assign, d_centroids);
    }

    // Copy final assignments back to the host
    cudaMemcpy(h_assign, d_assign, N * sizeof(int), cudaMemcpyDeviceToHost);


    // Print final clusters with assigned datapoints
    printf("Final Clusters:\n");
    for (int i = 0; i < N; i++) {
        printf("Datapoint Row %d assigned to Cluster %d\n", i, h_assign[i]);
    }

    // Write clustering number to compare the result
    FILE* inputFile = fopen("iris.csv", "r");
    // FILE* inputFile = fopen("winequality-red.csv", "r");
    if (inputFile == NULL) {
        fprintf(stderr, "Error opening iris file.");
        exit(EXIT_FAILURE);
    }

    FILE* result = fopen("result_iris.csv", "w");
    // FILE* result = fopen("result_wineQuality.csv", "w");
    if (result == NULL) {
        fprintf(stderr, "Error opening result file.");
        exit(EXIT_FAILURE);
    }

    int c = 0;
    char buffer[1024];
    while(fgets(buffer, sizeof(buffer), inputFile) != NULL) {
        // remove newline
        buffer[strcspn(buffer, "\n")] = '\0';

        // convert int value to string
        char intStr[K];
        sprintf(intStr, ",%d", h_assign[c]);

        // append to end of line
        strcat(buffer, intStr);
   
        // write to result file
        fprintf(result, "%s\n", buffer);
        c++;
    }

    // close files
    fclose(inputFile);
    fclose(result);

    // free host memory
    free(h_datapoints);
    free(h_centroids);
    free(h_assign);

    // Free device memory
    cudaFree(d_datapoints);
    cudaFree(d_centroids);
    cudaFree(d_assign);

    return 0;
}