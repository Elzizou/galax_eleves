//#define GALAX_MODEL_CPU_FAST
#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

#include <mipp.h>
#include <omp.h>
#include <math.h>
#include <iostream>

Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}

void Model_CPU_fast
::step()
{

	//int n = mipp::N<float>();
	//std::cout << "kek " << n << std::endl;


	std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
	std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
	std::fill(accelerationsz.begin(), accelerationsz.end(), 0);
	
	#pragma omp parallel for
	for (int i = 0; i < n_particles; i++)
	{
		for (int j = i+1; j < n_particles; j++)
		{
			const float diffx = particles.x[j] - particles.x[i];
			const float diffy = particles.y[j] - particles.y[i];
			const float diffz = particles.z[j] - particles.z[i];

			float dij = diffx * diffx + diffy * diffy + diffz * diffz;

			//float dijcond = static_cast<float>(signbit(dij-1))-0.5;
			//dij = (1.0+dijcond)*10.0/(dij*std::sqrt(dij)) + (1.0-dijcond)*10.0;

			if (dij < 1.0)
			{
				dij = 10.0;
			}
			else
			{
				dij = 10.0 / (dij * std::sqrt(dij));
			}

			//std::vector<float> v1(7);
			//std::vector<float> v2(7);

			float tmp = dij * initstate.masses[j];

			/*v1[0] = tmp;
			v1[1] = diffx;
			v1[2] = diffy;
			v1[3] = diffz;
			v1[4] = accelerationsx[i];
			v1[5] = accelerationsy[i];
			v1[6] = accelerationsz[i];*/

			mipp::Reg<float> r1,r2,r3,r4;

			r1 = tmp; 	//tmp
			r2 = {diffx, diffy, diffz, 0.};		// diffxyz
			r3 = {accelerationsx[i], accelerationsy[i], accelerationsz[i], 0.};		//accelerationsxyz
			r4 = mipp::fmadd(r1,r2,r3);

			accelerationsx[i] = r4[0];
			accelerationsy[i] = r4[1];
			accelerationsz[i] = r4[2];

			tmp = dij * initstate.masses[i];

			r1 = tmp;
			r3 = {-accelerationsx[j], -accelerationsy[j], -accelerationsz[j], 0.};
			r4 = mipp::fnmadd(r1,r2,r3);
			accelerationsx[j] = r4[0];
			accelerationsy[j] = r4[1];
			accelerationsz[j] = r4[2];
		}
	}

	
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

/**
 * #include <cmath>

#include "Model_CPU_naive.hpp"

Model_CPU_naive
::Model_CPU_naive(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}
//cc
void Model_CPU_naive
::step()
{
	std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
	std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
	std::fill(accelerationsz.begin(), accelerationsz.end(), 0);
	
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

				if (dij < 1.0)
				{
					dij = 10.0;
				}
				else
				{
					dij = std::sqrt(dij);
					dij = 10.0 / (dij * dij * dij);
				}

				accelerationsx[i] += diffx * dij * initstate.masses[j];
				accelerationsy[i] += diffy * dij * initstate.masses[j];
				accelerationsz[i] += diffz * dij * initstate.masses[j];
			}
		}
	}
	
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
}*/