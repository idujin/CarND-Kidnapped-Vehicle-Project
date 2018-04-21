/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 50;
	default_random_engine gen;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	weights.reserve(num_particles);

	for(int i=0; i< num_particles; ++i)
	{
		double sample_x, sample_y, sample_theta;

		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);

		Particle p;

		p.id= i;
		p.x = sample_x;
		p.y = sample_y;
		p.theta = sample_theta;
		p.weight = 1.0;

		particles.push_back(p);
		weights.push_back(1.0);

	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);
	for(int i = 0; i< num_particles; ++i)
	{
		if(abs(yaw_rate) < 0.001)
		{
			const double x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
			const double y = particles[i].y + velocity * delta_t * sin(particles[i].theta);

			particles[i].x = x;
			particles[i].y = y;
		}
		else
		{
			double x = particles[i].x + velocity/yaw_rate*(sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			double y = particles[i].y + velocity/yaw_rate*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			double theta = particles[i].theta + yaw_rate* delta_t;

			particles[i].x = x;
			particles[i].y = y;
			particles[i].theta = theta;

		}
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}

}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// find the predicted measurement
	for(int i =0; i < predicted.size(); ++i)
	{
		double pred_x = predicted[i].x;
		double pred_y = predicted[i].y;
		double min_dist = 100;
		int min_id =  0;


		for(int j=0; j < observations.size(); ++j)
		{
			double obs_x = observations[j].x;
			double obs_y = observations[j].y;

			double distance = dist(pred_x, pred_y, obs_x, obs_y);

			if(min_dist > distance || j == 0)
			{
				min_dist = distance;
				min_id = observations[j].id;
			}

		}
		predicted[i].id =  min_id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	const double sig_x = std_landmark[0];
	const double sig_y = std_landmark[1];
	// calculate normalization term
	const double gauss_norm= (1/(2 * M_PI * sig_x * sig_y));

	const int particleSize = particles.size();
	double weight_sum = 0.0;
	for(int i =0; i < particleSize; ++i)
	{
		
		const double x_part = particles[i].x;
		const double y_part = particles[i].y;
		const double theta = particles[i].theta;
		
		std::vector<Map::single_landmark_s> near_landmarks;
		near_landmarks.clear();

		for(auto& landmark: map_landmarks.landmark_list)
		{
			const double distance = dist(x_part, y_part, landmark.x_f, landmark.y_f);
			
			if(sensor_range >= distance)
			{
				near_landmarks.push_back(landmark);
				
			}
		}

		const int observationSize = observations.size();
		for(int j =0; j < observationSize; j++)
		{

			const double x_obs = observations[j].x;
			const double y_obs = observations[j].y;
			
			// transform to map x coordinate
			const double map_x = x_part + (cos(theta) * x_obs) - (sin(theta) * y_obs);
			// transform to map y coordinate
			const double map_y = y_part + (sin(theta) * x_obs) + (cos(theta) * y_obs);

			int min_id = -1;
			double min_dist = -1;
			for(auto& landmark : near_landmarks)
			{
				double distance = dist(map_x, map_y, landmark.x_f, landmark.y_f);
				if(min_id == -1 || (distance < min_dist))
				{
					min_dist = distance;
					min_id = landmark.id_i;

				}

			}
			const double mu_x = map_landmarks.landmark_list[min_id -1].x_f;
			const double mu_y = map_landmarks.landmark_list[min_id -1].y_f;
			const double diff_x = map_x - mu_x;
			const double diff_y = map_y - mu_y;			

			// calculate exponent
			const double exponent= (diff_x*diff_x)/(2 * sig_x*sig_x) + (diff_y*diff_y)/(2 * sig_y*sig_y);
			// calculate weight using normalization terms and exponent
			const double weight= gauss_norm * exp(-exponent);

			particles[i].weight *= weight;

			if (particles[i].weight == 0.0)
            {
                particles[i].weight = std::numeric_limits<double>::epsilon();
            }


		}
		weight_sum += particles[i].weight;


	}

	for(int i =0; i < particleSize; ++i)
	{
		double normWeight = particles[i].weight/ weight_sum;
		particles[i].weight = normWeight;
		weights[i] = normWeight;
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	// http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	std::discrete_distribution<> prob_list(weights.begin(), weights.end());

	std::vector<Particle> resampledParticles;
	const int particleSize = particles.size();

	for( int i = 0; i < particleSize; ++i)
	{
		resampledParticles.push_back(particles[prob_list(gen)]);
	}

	particles = resampledParticles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
