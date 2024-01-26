// Eden Forbes
// MinCogEco Predator

#include "Predator.h"
#include "Prey.h"
#include "random.h"
#include "CTRNN.h"

// Constants
const double SpaceSize = 5000;
const double HalfSpace = SpaceSize/2;

// *******
// Control
// *******

// Init the agent
void Predator::Set(int networksize, double pred_gain, double pred_s_width, double pred_frate, double pred_feff, double pred_metaloss, double pred_b_thresh, double pred_handling_time)
{
    size = networksize;
	gain = pred_gain; 
    sensorweights.SetBounds(1, 2*size);
	sensorweights.FillContents(0.0);
	pos = 0.0;
	pastpos = 0.0;
	sensor = 0.0;
    s_width = pred_s_width;
    state = 1.0;
    frate = pred_frate;
    feff = pred_feff;
    metaloss = pred_metaloss;
    breathing_rate = 0.0;
    death = false;
    birth = false;
    birth_thresh = pred_b_thresh;
    s_scalar = 0.0;
    a_scalar = 0.0;
    handling_time = pred_handling_time;
    handling_counter = 0.0;
    handling = false;
    // interaction rates
    munchrate = 0.0;
    birthrate = 0.0;
    snackflag = 0.0;
}

// Reset the state of the agent
void Predator::Reset(double initpos, double initstate)
{
	pos = initpos;
	pastpos = initpos;
	sensor = 0.0;
    state = initstate;
    death = false;
    birth = false;
	NervousSystem.RandomizeCircuitState(0.0,0.0);
    handling = false;
    handling_counter = 0.0;
}

// Map the output of neuron 3 to the breathing rate
double Predator::MapBreathingRate(double neuronOutput){
    double minRate = 0.5;
    double maxRate = 2.0;

    return minRate + (maxRate - minRate) * neuronOutput;

}

// Sniff Sense 
void Predator::Sense(TVector<double> &prey_scent, double timestep)
{
    // Get Breathing Rate
    double bRate = MapBreathingRate(NervousSystem.NeuronOutput(3));
    double phase = sin(timestep * 2 * M_PI * bRate);
    breathing_rate = bRate;
    // Get Left Concentrations
    double pc_left = 0.0;

    // Get Right Concentrations
    double pc_right = 0.0;

    if (phase > 0) {
        // Left side
        int first_left = floor(pos);
        for (int i = 0; i <= s_width; i++){
            double l_loc = first_left - i;
            if (l_loc < 0){
                l_loc = l_loc + SpaceSize;
            }
            pc_left +=(-2*exp((-i * i) / (2 * s_width * s_width))) * prey_scent[l_loc];
        }
        // Right side
        int first_right = ceil(pos);
        for (int i = 0; i <= s_width; i++){
            double r_loc = first_right + i;
            if (r_loc > SpaceSize){
                r_loc = r_loc - SpaceSize;
            }
            pc_right +=(2*exp((-i * i) / (2 * s_width * s_width))) * prey_scent[r_loc];
        }
        // Augment depending on BR/phase
        pc_left = pc_left * phase;
        pc_right = pc_right * phase;
        sensor = pc_left + pc_right;
    }
    // If phase < 0, all sensory inputs will be 0
}

// Step
void Predator::Step(double StepSize, TVector<double> &WorldFood, TVector<Prey> &preylist)
{
    // Remember past position
    pastpos = pos;
	// Update the body position based on the other 2 neurons
    // If still handling previous catch, don't move
    if (handling == true){
        handling_counter += 1;
        if (handling_counter >= handling_time){
            handling = false;
            handling_counter = 0;
        }
    }
    else{
        double N1IP = sensor*sensorweights[1] + state*sensorweights[2];
        double N2IP = sensor*sensorweights[3] + state*sensorweights[4];
        double N3IP = sensor*sensorweights[5] + state*sensorweights[6];
        // Give each interneuron its sensory input
        NervousSystem.SetNeuronExternalInput(1, N1IP);
        NervousSystem.SetNeuronExternalInput(2, N2IP);
        NervousSystem.SetNeuronExternalInput(3, N3IP);
        // Update the nervous system
        NervousSystem.EulerStep(StepSize);
        pos += StepSize * gain * (NervousSystem.NeuronOutput(2) - NervousSystem.NeuronOutput(1));
        // Update State if the agent passed food
        if (pastpos < pos){
            if (pos > WorldFood.Size()){
                pos = pos - WorldFood.Size();
                for (int i = 0; i < preylist.Size(); i++){
                    if (preylist[i].pos > pastpos && preylist[i].pos <= WorldFood.Size()){
                        preylist[i].state -= preylist[i].state*frate;
                        state += preylist[i].state*feff;
                        snackflag += 1;
                        handling = true;
                    }
                    else if (preylist[i].pos >= 0 && preylist[i].pos < pos){
                        preylist[i].state -= preylist[i].state*frate;
                        state += preylist[i].state*feff;
                        snackflag += 1;
                        handling = true;
                    }
                }
            }
            else {
                for (int i = 0; i < preylist.Size(); i++){
                    if (preylist[i].pos > pastpos && preylist[i].pos <= pos){
                        preylist[i].state -= preylist[i].state*frate;
                        state += preylist[i].state*feff;
                        snackflag += 1;
                        handling = true;
                    }
                }
            }
        }
        if (pastpos > pos){
            if (pos < 0){
                pos = pos + WorldFood.Size();
                for (int i = 0; i < preylist.Size(); i++){
                    if (preylist[i].pos < pastpos && preylist[i].pos >= 0){
                        preylist[i].state -= preylist[i].state*frate;
                        state += preylist[i].state*feff;
                        snackflag += 1;
                        handling = true;
                    }
                    else if (preylist[i].pos <= WorldFood.Size() && preylist[i].pos > pos){
                        preylist[i].state -= preylist[i].state*frate;
                        state += preylist[i].state*feff;
                        snackflag += 1;
                        handling = true;
                    }
                }
            }
            if (pos >= 0) {
                for (int i = 0; i < preylist.Size(); i++){
                    if (preylist[i].pos < pastpos && preylist[i].pos >= pos){
                        preylist[i].state -= preylist[i].state*frate;
                        state += preylist[i].state*feff;
                        snackflag += 1;
                        handling = true;
                    }
                }
            }
        }
    }
    // Lose state over time
    // Physiological one needs fixing
    // state -= metaloss * (pow(s_scalar,2)/s_width) * (pow(a_scalar,2)/gain);
    state -= metaloss * breathing_rate;

    // Birth & Death
    if (state <= 0){
        death = true;
    }
    if (state > birth_thresh){
        birth = true;
    }
}


// // Sense 
// void Predator::Sense(TVector<double> &prey_loc)
// {
//     // Sense
// 	double mindistL = 99999;
//     double mindistR = 99999;
//     for (int i = 0; i < prey_loc.Size(); i++){
//         double d = prey_loc[i] - pos;
//         // printf("d = %f\n", d);
//         if (d < 0 && d >= -HalfSpace){
//             // Closest to the left side, distance is as calculated
//             if (abs(d) < mindistL){
//                 mindistL = abs(d);
//             }
//         }
//         else if (d < 0 && d < -HalfSpace){
//             // Closest to the right side, distance is total area + - left side distance 
//             d = SpaceSize + d;
//             if (d < mindistR){
//                 mindistR = d;
//             }
//         }
//         else if (d > 0 && d > HalfSpace){ 
//             // Closest to the left side, distance is -total area + right side distance
//             d = -SpaceSize + d;
//             if (abs(d) < mindistL){
//                 mindistL = abs(d);
//             }
//         }
//         else if (d > 0 && d <= HalfSpace){
//             // Closest to the right side, distance is as calculated
//             if (d < mindistR){
//                 mindistR = d;
//             }
//         }
//         else if (d == 0){
//             d = 0;
//         }
//         else{
//             printf("Prey pos size = %d\n", prey_loc.Size());
//             printf("Error in predator sensing\n");
//             printf("d = %f\n", d);
//         }
//     }
//     // Cumulate, distance fits in the Gaussian as the difference of the mean (position) and the state (food position)
//     // Negate left so negative sensor reading says food left, positive says food right, zero says no food or food on both sides.
// 	sensor = -2*exp(-(mindistL) * (mindistL) / (2 * s_width * s_width)) + 2*exp(-(mindistR) * (mindistR) / (2 * s_width * s_width));
//     // printf("Pred sensor = %f\n", sensor);
// }