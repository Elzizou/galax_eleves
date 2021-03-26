//#define GALAX_MODEL_CPU_FAST
#ifdef GALAX_MODEL_CPU_FAST

/**
 * TODO : Gérer les cas ou n_particules n'est pas divisible par n_reg
 * optimiser le for avec les autres opti
 **/


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
	constexpr int n_reg = mipp::N<float>();
	
	#pragma omp parallel for schedule(static,1)
	for (int i = 0; i < n_particles; i++)
	{
		for (int j = 0; j < n_particles-n_reg; j+=n_reg)
		{		
			if(i != j)
			{
				mipp::Reg<float> vdiffx, vdiffy, vdiffz, vdij, vtmp, vtmp1, vaccelx, vaccely, vaccelz;

					vdiffx = &particles.x[j];
					vdiffx = vdiffx-particles.x[i];

					vdiffy = &particles.y[j];
					vdiffy = vdiffy-particles.y[i];

					vdiffz = &particles.z[j];
					vdiffz = vdiffz-particles.z[i];

					vtmp = vdiffz*vdiffz;
					vdij = fmadd(vdiffy, vdiffy, vtmp);
					vdij = fmadd(vdiffx, vdiffx, vdij);
					
					vtmp  = 10.;
					vdij  = vtmp*mipp::rsqrt(vdij)/vdij;
					vdij  = mipp::min(vdij, vtmp);

					vtmp1 = initstate.masses[i];
					vtmp = vdij*vtmp1;
					vaccelx = &accelerationsx[j];
					vaccely = &accelerationsy[j];
					vaccelz = &accelerationsz[j];

					vaccelx = fnmadd(vtmp, vdiffx, vaccelx);
					vaccely = fnmadd(vtmp, vdiffy, vaccely);
					vaccelz = fnmadd(vtmp, vdiffz, vaccelz);

					vaccelx.store(&accelerationsx[j]);
					vaccely.store(&accelerationsy[j]);
					vaccelz.store(&accelerationsz[j]);
			}
		}
		
		for (int j = n_particles-n_reg; j < n_particles; j++)
		{		
			if(i != j)
			{
			const float diffx = particles.x[j] - particles.x[i];
			const float diffy = particles.y[j] - particles.y[i];
			const float diffz = particles.z[j] - particles.z[i];

			float dij = diffx * diffx + diffy * diffy + diffz * diffz;

			//float dijcond = static_cast<float>(signbit(dij-1))-0.5;
			//dij = (1.0+dijcond)*10.0/(dij*std::sqrt(dij)) + (1.0-dijcond)*10.0;

			dij = fmin(10.0, dij = 10.0/(std::sqrt(dij) * dij));
			//int n = mipp:N<float>(); keke


			float tmp = dij * initstate.masses[j];
			accelerationsx[i] += diffx * tmp;
			accelerationsy[i] += diffy * tmp;
			accelerationsz[i] += diffz * tmp;
			}
		}
	}

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