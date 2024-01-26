#pragma once

#include "CTRNN.h"

// The Sniffer Agent class declaration
class Sniffer {
public:
    // The constructor
    Sniffer(int networksize);

    // The destructor
   	~Sniffer();

    // Accessors
    double GetPosX() const { return posX; }
    double GetPosY() const { return posY; }
    double GetPastPosX() const { return pastposX; }
    double GetPastPosY() const { return pastposY; }
    double GetVelocity() const { return velocity; }
    double GetTheta() const { return theta; }
    void SetPosX(double newPosX) { posX = newPosX; }
    void SetPosY(double newPosY) { posY = newPosY; }
    void SetSensorWeight(int index, double value) { sensorweights[index] = value; }
    double GetBreathingRate() {return MapBreathingRate(NervousSystem.NeuronOutput(3));}
    double GetCO2Level(){return co2Level;}
    double GetOxygenLevel(){return oxygenLevel;}
    double GetPassedOutState() const {return is_passed_out;}
    
    // Control methods
    void Set(int networksize);
    void Reset(double initposX, double initposY, double initTheta);
    double MapBreathingRate(double neuronOutput);
    // void Sense(double chemical_concentration, double current_time);
    // void Sense(double leftConcentration, double rightConcentration);
    void Sense(double leftConcentration, double rightConcentration, double current_time);
    void Step(double StepSize);
    void Respirate(double StepSize);
    double CalculateRespiratoryState();
    double dCO2dt(double R);
    double dO2dt(double R);
    // Properties
    int size;
    double breathingRate;
    bool is_passed_out;
    double posX, posY, pastposX, pastposY, velocity, theta, pastTheta, gain, leftSensor, rightSensor, sensor, oxygenLevel, co2Level;
    TVector<double> sensorweights;
    CTRNN NervousSystem;
};