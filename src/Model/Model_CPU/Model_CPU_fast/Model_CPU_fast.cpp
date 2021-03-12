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
	int n_reg = mipp::N<float>();
	
	#pragma omp parallel for
	for (int i = 0; i < n_particles; i++)
	{
		for (int j = i+1; j < n_particles/n_reg; j++)
		{		
			mipp::Reg<float> diffx, diffy, diffz, dij,
							tmp, accelx, accely, accelz, dijcond, tmp1;
			//const float diffx = particles.x[j] - particles.x[i];
			//const float diffy = particles.y[j] - particles.y[i];
			//const float diffz = particles.z[j] - particles.z[i];

			/*for(int k = 0; k<n_reg; k++)
			{
				diffx[k] = particles.x[j+k] - particles.x[i+k];
				diffy[k] = particles.y[j+k] - particles.y[i+k];
				diffz[k] = particles.z[j+k] - particles.z[i+k];
			}*/

			tmp   = &particles.x[i];
			diffx = &particles.x[j];
			diffx = diffx-tmp;

			tmp   = &particles.y[i];
			diffy = &particles.y[j];
			diffy = diffy-tmp;

			tmp   = &particles.z[i];
			diffz = &particles.z[j];
			diffz = diffz-tmp;

			//dij = diffx * diffx + diffy * diffy + diffz * diffz;
			tmp = diffz*diffz;
			dij = fmadd(diffy, diffy, tmp);
			dij = fmadd(diffx, diffx, dij); // dij = diffx^2+diffy^2+diffz^2

			tmp1 = 10.;
			dij = tmp1*mipp::rsqrt(dij)/dij;
			dij = max(dij, tmp1);

			/*if (dij < 1.0)
			{
				dij = 10.0;
			}
			else
			{
				dij = 10.0 / (dij * std::sqrt(dij));
			}*/

			//int n = mipp:N<float>();


			/*for(int k = 0; k<n_reg; k++)
			{
				tmp[k] = dij[k] * initstate.masses[i+k];
				accelx[i] = accelerationsx[i+k];
				accely[i] = accelerationsy[i+k];
				accelz[i] = accelerationsz[i+k];
			}*/

			tmp1 = &initstate.masses[j];
			tmp  = dij*tmp1;
			accelx = &accelerationsx[i];
			accely = &accelerationsy[i];
			accelz = &accelerationsz[i];

			accelx = fmadd(tmp, diffx, accelx);
			accely = fmadd(tmp, diffy, accely);
			accelz = fmadd(tmp, diffz, accelz);

			/*for(int k = 0; k<n_reg; k++)
			{
				accelerationsx[i+k] = accelx;
				accelerationsy[i+k] = accely;
				accelerationsz[i+k] = accelz;
			}*/

			accelx.store(&accelerationsx[i]);
			accely.store(&accelerationsy[i]);
			accelz.store(&accelerationsz[i]);


			/*for(int k = 0; k<n_reg; k++)
			{
				tmp[k] = dij[k] * initstate.masses[j+k];
				accelx[k] = -accelerationsx[j+k];
				accely[k] = -accelerationsy[j+k];
				accelz[k] = -accelerationsz[j+k];
			}*/
			tmp1 = &initstate.masses[i];
			tmp = dij*tmp1;
			accelx = &accelerationsx[j];
			accely = &accelerationsy[j];
			accelz = &accelerationsz[j];

			accelx = fnmadd(tmp, diffx, accelx);
			accely = fnmadd(tmp, diffy, accely);
			accelz = fnmadd(tmp, diffz, accelz);
			

			/*for(int k = 0; k<n_reg; k++)
			{
				accelerationsx[j+k] = accelx;
				accelerationsy[j+k] = accely;
				accelerationsz[j+k] = accelz;
			}*/
			accelx.store(&accelerationsx[j]);
			accely.store(&accelerationsy[j]);
			accelz.store(&accelerationsz[j]);
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