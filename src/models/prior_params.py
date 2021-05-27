prior_params = {
    
    "sigma": 0.14,  # std [m/100ms]
    "sigma_eps": 0.05,  # std of noise measurement [m]

    # Wifi signal strength priors
    "mu_omega_0": -45.0,
    # Signal strength uncertainty
    "sigma_omega_0": 10.0,

    # How accurate is the measured signal stregth
    "sigma_omega": 3.0,

    # Beacon uncertainty
    "sigma_delta": 10.
    
}
