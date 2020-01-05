#include "cuda_helper.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

#include <random>
#include <iostream>

static constexpr int BLOCK_SIZE = 128;
static constexpr float SOFTENING = 1e-9f;
static constexpr float TIMESTEP = 0.0001f; // time step
static constexpr int ITERATIONS = 10;  // simulation iterations
static constexpr float DELTA = ITERATIONS*0.0001; // for floating-point error

struct Body {
  float x, y, z, vx, vy, vz;
};

// error checker
// (nbody with single-precision parallel Euler is not that robust, small timestep needed)
bool compare_equal_pos(Body* p1, Body* p2, int nbodies) {
  for(int i=0; i<nbodies; ++i) {
    float dx = p1[i].x - p2[i].x;
    float dy = p1[i].y - p2[i].y;
    float dz = p1[i].z - p2[i].z;
    if(  std::abs(dx) > DELTA
      || std::abs(dy) > DELTA
      || std::abs(dz) > DELTA ) {
      std::cerr << "First mismatch"
                << " at [" << i << "]"
                << ": (" << dx <<","<< dy <<","<< dz <<")\n"
                << ": (" << p1[i].y <<","<< p2[i].y <<")\n"
        ;
      return false;
    }
  }
  return true;
}

void bodyCPU(Body* p, int nbodies) {
  for (int iter = 1; iter <= ITERATIONS; iter++) {
    // force
    for(int i=0; i<nbodies; ++i) {
      float Fx = 0.0f;
      float Fy = 0.0f;
      float Fz = 0.0f;

      for(int j=0; j<nbodies; ++j) {
        float dx = p[j].x - p[i].x;
        float dy = p[j].y - p[i].y;
        float dz = p[j].z - p[i].z;
        float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        float invDist = 1.0/sqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
      }
      p[i].vx += TIMESTEP*Fx;
      p[i].vy += TIMESTEP*Fy;
      p[i].vz += TIMESTEP*Fz;
    }
    // integrate
    for(int i=0; i<nbodies; ++i) {
      p[i].x += p[i].vx*TIMESTEP;
      p[i].y += p[i].vy*TIMESTEP;
      p[i].z += p[i].vz*TIMESTEP;
    }
  }
}

// Kernel
__global__
void bodyIntegrate(Body* p, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < n) {
    p[i].x += p[i].vx*TIMESTEP;
    p[i].y += p[i].vy*TIMESTEP;
    p[i].z += p[i].vz*TIMESTEP;
  }
}

__global__
void bodyForce(Body* p, int n) {

  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      // 3 FLOPS
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      // 6 FLOPS
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      // 2 FLOPS (1 sqrt, 1 inv)
      float invDist = 1.0/sqrtf(distSqr);
      // 2 FLOPS
      float invDist3 = invDist * invDist * invDist;

      // 6 FLOPS
      Fx += dx * invDist3;
      Fy += dy * invDist3;
      Fz += dz * invDist3;
    }

    p[i].vx += TIMESTEP*Fx;
    p[i].vy += TIMESTEP*Fy;
    p[i].vz += TIMESTEP*Fz;
  }
}


// Host

int main(const int argc, const char** argv) {

  int nbodies = 30000;
  int dev = 0;
  if (argc > 1)
    nbodies = atoi(argv[1]);
  if (argc > 2)
    dev = atoi(argv[2]);

  std::cout << "USAGE\n ./nbody [nbodies] [device-index]\n\n";
  dim3 blocks( (nbodies-1)/BLOCK_SIZE+1 );
  // Device information
  CHECK_CUDA( cudaSetDevice(dev) );
  std::cout << getCUDADeviceInformations(dev).str()
            << "\n"
            << "\nThreads per block: "<< BLOCK_SIZE
            << "\nBlocks per SM: " << blocks.x << " (monolithic)"
            << "\nEpsilon: " << SOFTENING
            << "\nTimestep: " << TIMESTEP
            << "\nIterations: " << ITERATIONS
            << "\nDelta: " << DELTA
            << "\n\n"
    ;

  // for time measurement
  float milliseconds = 0;
  float min_ms = std::numeric_limits<float>::max();
  cudaEvent_t cstart, cend;
  CHECK_CUDA(cudaEventCreate(&cstart));
  CHECK_CUDA(cudaEventCreate(&cend));

  //
  Body* p;
  Body* d_p;
  p = new Body[nbodies];
  CHECK_CUDA( cudaMalloc(&d_p, nbodies*sizeof(Body)) );

  // C++11 random generator for uniformly distributed numbers in {1,..,42}, w seed
  std::default_random_engine eng{ 1337 };
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  for(int i=0; i<nbodies; ++i) {
    p[i].x  = dist(eng);
    p[i].y  = dist(eng);
    p[i].z  = dist(eng);
    p[i].vx = dist(eng);
    p[i].vy = dist(eng);
    p[i].vz = dist(eng);
  }

  CHECK_CUDA(cudaMemcpy(d_p, p, nbodies*sizeof(Body), cudaMemcpyHostToDevice));

  // benchmark loop
  for (int iter = 1; iter <= ITERATIONS; iter++) {
    CHECK_CUDA(cudaEventRecord(cstart));

    bodyForce<<<blocks, BLOCK_SIZE>>>(d_p, nbodies);

    CHECK_CUDA( cudaEventRecord(cend) );
    CHECK_CUDA( cudaEventSynchronize(cend) );
    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaEventElapsedTime(&milliseconds, cstart, cend) );
    if(milliseconds<min_ms)
      min_ms = milliseconds;

    // no time measurement of integration
    bodyIntegrate<<<blocks, BLOCK_SIZE>>>(d_p, nbodies);
    CHECK_CUDA( cudaGetLastError() );
  }

  // validate results if n<8193 (CPU is so slow)
  if(nbodies<8193) {
    Body* p2 = new Body[nbodies];
    bodyCPU(p, nbodies);

    CHECK_CUDA(cudaMemcpy(p2, d_p, nbodies*sizeof(Body), cudaMemcpyDeviceToHost));
    if(compare_equal_pos(p, p2, nbodies)) {
      std::cout << "SUCCESS\n";
    } else {
      std::cout << "FAILED\n";
    }
    delete[] p2;
  } else {
    std::cout << "No validation.\n";
  }

  std::cout << "Bodies: " << nbodies
            << "\nGFLOPs: " << 19.0 * 1e-6 * nbodies * nbodies / min_ms
            << "\n";

  CHECK_CUDA(cudaEventDestroy(cstart));
  CHECK_CUDA(cudaEventDestroy(cend));
  delete[] p;
  CHECK_CUDA( cudaFree( d_p ) );

  CHECK_CUDA(cudaDeviceReset());
  return 0;
}
