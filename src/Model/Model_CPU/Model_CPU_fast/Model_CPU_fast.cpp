#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

#include <algorithm>
#include <mipp.h>
#include <omp.h>
#include <math.h>

#define SCHEDULED 0
#define COLLAPSED 1
#define REDUCTION 2
#define COLLAPSED_SCHEDULED 3
#define FOR 4
#define FOR_TRIG 5
#define STRATEGY FOR


Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}

#if STRATEGY == REDUCTION
#endif

void Model_CPU_fast
::step()
{
	std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
	std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
	std::fill(accelerationsz.begin(), accelerationsz.end(), 0);
#if STRATEGY == SCHEDULED
	#pragma omp parallel for schedule(static,1)
	for (int i = 0; i < n_particles; i++)
	{
		for (int j = i+1; j < n_particles; j++)
		{
			const float diffx = particles.x[j] - particles.x[i];
			const float diffy = particles.y[j] - particles.y[i];
			const float diffz = particles.z[j] - particles.z[i];

			float dij = diffx * diffx + diffy * diffy + diffz * diffz;
			dij = fmin(10.0, dij = 10.0/(std::sqrt(dij) * dij));
			//float dijcond = static_cast<float>(signbit(dij-1))-0.5;
			//dij = (1.0+dijcond)*10.0/(dij*std::sqrt(dij)) + (1.0-dijcond)*10.0;

			// if (dij < 1.0)
			// {
			// 	dij = 10.0;
			// }
			// else
			// {
			// 	dij = 10.0 / (dij * std::sqrt(dij));
			// }

			//int n = mipp:N<float>();


			float tmp = dij * initstate.masses[j];
			accelerationsx[i] += diffx * tmp;
			accelerationsy[i] += diffy * tmp;
			accelerationsz[i] += diffz * tmp;

			tmp = dij * initstate.masses[i];
			accelerationsx[j] -= diffx * tmp;
			accelerationsy[j] -= diffy * tmp;
			accelerationsz[j] -= diffz * tmp;
		}
	}
#elif STRATEGY == COLLAPSED
	#pragma omp parallel for collapse(2) 
	for (int i = 0; i < n_particles; i++)
	{
		for (int j = 0; j < n_particles; j++)
		{
			if(i != j)
			{
				const float diffx = particles.x[j] - particles.x[i];
				const float diffy = particles.y[j] - particles.y[i];
				const float diffz = particles.z[j] - particles.z[i];

				float dij = diffx * diffx + diffy * diffy + diffz * diffz;
				dij = fmin(10.0, dij = 10.0/(std::sqrt(dij) * dij));
				// if (dij < 1.0)
				// {
				// 	dij = 10.0;
				// }
				// else
				// {
				// 	dij = std::sqrt(dij);
				// 	dij = 10.0 / (dij * dij * dij);
				// }

				accelerationsx[i] += diffx * dij * initstate.masses[j];
				accelerationsy[i] += diffy * dij * initstate.masses[j];
				accelerationsz[i] += diffz * dij * initstate.masses[j];
			}
		}
	}
#elif STRATEGY == REDUCTION
	#pragma omp parallel for schedule(static, 1) reduction()
	for (int i = 0; i < n_particles; i++)
	{
		for (int j = 0; j < n_particles; j++)
		{
			if(i != j)
			{
				const float diffx = particles.x[j] - particles.x[i];
				const float diffy = particles.y[j] - particles.y[i];
				const float diffz = particles.z[j] - particles.z[i];

				float dij = diffx * diffx + diffy * diffy + diffz * diffz;
				dij = fmin(10.0, dij = 10.0/(std::sqrt(dij) * dij));
				// if (dij < 1.0)
				// {
				// 	dij = 10.0;
				// }
				// else
				// {
				// 	dij = std::sqrt(dij);
				// 	dij = 10.0 / (dij * dij * dij);
				// }

				accelerationsx[i] += diffx * dij * initstate.masses[j];
				accelerationsy[i] += diffy * dij * initstate.masses[j];
				accelerationsz[i] += diffz * dij * initstate.masses[j];
			}
		}
	}
#elif STRATEGY == COLLAPSED_SCHEDULED
	#pragma omp parallel for simd collapse(2) schedule(auto)
	for (int i = 0; i < n_particles; i++)
	{
		for (int j = 0; j < n_particles; j++)
		{
			if(i != j)
			{
				const float diffx = particles.x[j] - particles.x[i];
				const float diffy = particles.y[j] - particles.y[i];
				const float diffz = particles.z[j] - particles.z[i];

				float dij = diffx * diffx + diffy * diffy + diffz * diffz;
				dij = fmin(10.0, dij = 10.0/(std::sqrt(dij) * dij));
				// if (dij < 1.0)
				// {
				// 	dij = 10.0;
				// }
				// else
				// {
				// 	dij = std::sqrt(dij);
				// 	dij = 10.0 / (dij * dij * dij);
				// }

				accelerationsx[i] += diffx * dij * initstate.masses[j];
				accelerationsy[i] += diffy * dij * initstate.masses[j];
				accelerationsz[i] += diffz * dij * initstate.masses[j];
			}
		}
	}
#elif STRATEGY == FOR
	#pragma omp parallel for
	for (int i = 0; i < n_particles; i++)
	{
		for (int j = 0; j < n_particles; j++)
		{
			if(i != j)
			{
				const float diffx = particles.x[j] - particles.x[i];
				const float diffy = particles.y[j] - particles.y[i];
				const float diffz = particles.z[j] - particles.z[i];

				float dij = diffx * diffx + diffy * diffy + diffz * diffz;
				dij = fmin(10.0, dij = 10.0/(std::sqrt(dij) * dij));
				// if (dij < 1.0)
				// {
				// 	dij = 10.0;
				// }
				// else
				// {
				// 	dij = std::sqrt(dij);
				// 	dij = 10.0 / (dij * dij * dij);
				// }

				accelerationsx[i] += diffx * dij * initstate.masses[j];
				accelerationsy[i] += diffy * dij * initstate.masses[j];
				accelerationsz[i] += diffz * dij * initstate.masses[j];
			}
		}
	}
#elif STRATEGY == FOR_TRIG
	#pragma omp parallel for 
	for (int i = 0; i < n_particles; i++)
	{
		for (int j = i+1; j < n_particles; j++)
		{
			const float diffx = particles.x[j] - particles.x[i];
			const float diffy = particles.y[j] - particles.y[i];
			const float diffz = particles.z[j] - particles.z[i];

			float dij = diffx * diffx + diffy * diffy + diffz * diffz;
			dij = fmin(10.0, dij = 10.0/(std::sqrt(dij) * dij));
			//float dijcond = static_cast<float>(signbit(dij-1))-0.5;
			//dij = (1.0+dijcond)*10.0/(dij*std::sqrt(dij)) + (1.0-dijcond)*10.0;

			// if (dij < 1.0)
			// {
			// 	dij = 10.0;
			// }
			// else
			// {
			// 	dij = 10.0 / (dij * std::sqrt(dij));
			// }

			//int n = mipp:N<float>();


			float tmp = dij * initstate.masses[j];
			accelerationsx[i] += diffx * tmp;
			accelerationsy[i] += diffy * tmp;
			accelerationsz[i] += diffz * tmp;

			tmp = dij * initstate.masses[i];
			accelerationsx[j] -= diffx * tmp;
			accelerationsy[j] -= diffy * tmp;
			accelerationsz[j] -= diffz * tmp;
		}
	}
#endif

#pragma omp parallel for
for (int i = 0; i < n_particles; i++)
{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
		velocitiesy[i] += accelerationsy[i] * 2.0f;
		velocitiesz[i] += accelerationsz[i] * 2.0f;
		particles.x[i] += velocitiesx   [i] * 0.1f;
		particles.y[i] += velocitiesy   [i] * 0.1f;
		particles.z[i] += velocitiesz   [i] * 0.1f;
}


// OMP  version
// #pragma omp parallel for
//     for (int i = 0; i < n_particles; i += mipp::N<float>())
//     {
//     }


// OMP + MIPP version
// #pragma omp parallel for
//     for (int i = 0; i < n_particles; i += mipp::N<float>())
//     {
//         // load registers body i
//         const mipp::Reg<float> rposx_i = &particles.x[i];
//         const mipp::Reg<float> rposy_i = &particles.y[i];
//         const mipp::Reg<float> rposz_i = &particles.z[i];
//               mipp::Reg<float> raccx_i = &accelerationsx[i];
//               mipp::Reg<float> raccy_i = &accelerationsy[i];
//               mipp::Reg<float> raccz_i = &accelerationsz[i];
//     }
}

#endif // GALAX_MODEL_CPU_FAST
