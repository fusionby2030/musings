import numpy as np 
# Constants
mu_0 = 4 * np.pi * 10 ** -7
m_i = 3.3435860e-27  # Mass of an ion of deuterium (kg) kg
m_p = 1.6726219e-27  # Mass of a proton (kg) 
m_e = 9.10938356e-31  # Mass of an electron (kg)


def calculate_alpha(pressure: np.ndarray, psi: np.ndarray, total_volume: float, major_radius: float, ) -> np.ndarray:
    """Approximation of alpha as defined in eq. 3 of Frassinetti. et al., 
    
    Approximations: 
    volume profile: parabolic via V = total_volume*psi**2

    Parameters
    ----------
    pressure : np.ndarray
        _description_
    total_volume : float
        _description_
    major_radius : float
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    V_psi = lambda x : total_volume*(x)**2 
    volume = V_psi(psi)

    c1 = 2*np.gradient(volume) / ((2*np.pi)**2)
    c2 = (volume / (2*np.pi*np.pi*major_radius))**(1/2)
    grad_pressure = np.gradient(pressure)
    return -c1*c2*grad_pressure# *mu_0


def calculate_boostrap_current(pressure: np.ndarray, temperature: np.ndarray, density: np.ndarray, psi: np.ndarray,
                                major_radius: float, minor_radius: float, q_95: float, toroidal_field_mag_axis: float) -> np.ndarray:
    
    """
    Approximate the bootstrap current using electron profiles and first order approximations of i) q and Bt profiles (equilibrium flux function) ii) collisionality.
    This is using eq. 14.12 given in Wesson 2nd Ed. 

    Approximations: 
    q profile: q(psi) = a*b^(psi)**3 where a = 1.1, b = ((q_95 / 1.1)**(1.0 / (0.95)))**3
    Bt profile: toroidal_field_mag_axis*np.exp(-psi / 1.0)
    x : ratio of trapped to circulating particles ~sqrt(2*inverse_aspectratio)
    collisionality (electron/ion) given by eqs. 2.15.1/2.15.2 for a single ionizied species

    n_e = n_i, t_e = t_i 
    Args:
        pressure (np.ndarray): 1D numpy array of pressure profile in Pascals.
        temperature (np.ndarray): 1D numpy array of temperature profile in Kelvin.
        density (np.ndarray): 1D numpy array of density profile in m^-3.
        psi (np.ndarray): 1D numpy array of radial coordinates in normalized flux coordinates.
        major_radius (float): Major radius of the plasma in meters.
        minor_radius (float): Minor radius of the plasma in meters.
        q (float): Safety factor.
        toroidal_field_mag_axis (float): Magnitude of the toroidal magnetic field at the magnetic axis in Tesla.

    Returns:
        np.ndarray: 1D numpy array of bootstrap current profile in Amperes.
    """

    # inverse aspect ratio from major and minor radius 
    epsilon = minor_radius / major_radius

    # ratio of trapped to circulating particles 
    x = np.sqrt(2) * np.sqrt(epsilon) 

    # First order approximation of torodial field as function of psi 
    # B_phi_psi = lambda x: (x + 1 / np.sqrt(toroidal_field_mag_axis))**(-2) 
    B_phi_psi = lambda x: 1 / ( x + 1 / toroidal_field_mag_axis)
    # toroidal_field = toroidal_field_mag_axis*np.exp(-psi) 
    toroidal_field = B_phi_psi(psi)
    f_psi = major_radius*toroidal_field / mu_0 # Equilibrium flux function using above 

    # First order approximation of q as function of psi -> parabolic
    n = 3
    a = 1.1
    b = ((q_95 / 1.1)**(1.0 / (0.95))**n)
    q_psi = lambda x: a*(b**(x**n))
    q = q_psi(psi)
    # collisionality
    omega_b_i = ((epsilon)**(1/2) * (temperature  / m_i)**(1/2)) / (major_radius*q)
    omega_b_e = ((epsilon)**(1/2) * (temperature  / m_e)**(1/2)) / (major_radius*q)

    couloumb_logarithm_e = 17.0
    couloumb_logarithm_i = couloumb_logarithm_e*1.1

    tau_e = 6.4 * 10 ** 14 * ((temperature / 1000.0) ** (3 / 2) / density) # formula given in keV
    tau_i = 6.6 * 10 ** 17 * np.sqrt(m_i / m_p) * ((temperature / 1000.0) ** (3 / 2) / (density * couloumb_logarithm_i)) # formula given in keV
    nu_e = 1 / tau_e
    nu_i = 1 / tau_i
    nu_star_e = nu_e  / (epsilon*omega_b_e)
    nu_star_i = nu_i / (epsilon*omega_b_i)

    # random fit thing 
    d = -(1.17) / (1 + 0.46*x)

    # calculate coefficeints c1, c2, c3, c4
    c1 = (4.0 + 2.6 * x) / ((1 + 1.02 * np.sqrt(nu_star_e) + 1.07 * nu_star_e) * (1 + 1.07 * epsilon ** (3 / 2) * nu_star_e))
    c2 = c1
    c3 = (7.0 + 6.5 * x) / ((1 + 0.57 * np.sqrt(nu_star_e) + 0.61 * nu_star_e) * (1 + 0.61 * epsilon ** (3 / 2) * nu_star_e)) - (5.0 / 2) * c1
    c4 = ((d + 0.35 * np.sqrt(nu_star_i)) / (1 + 0.7 * np.sqrt(nu_star_i)) + 2.1 * epsilon ** 3 * nu_star_i ** 2) / ((1 + epsilon ** 3 * nu_star_i ** 2) * (1 + epsilon ** 3 * nu_star_e ** 2)) * c2

    D_x = 2.4 + 5.4 * x + 2.6 * x ** 2

    grad_p_e = np.gradient(pressure)
    grad_T_e = np.gradient(temperature)
    # put it all together 
    boostrap_current = -(mu_0 * x * f_psi * pressure) / D_x * (2 * (c1 + c2) * grad_p_e / pressure + 2 * (c3 + c4) * grad_T_e / temperature)
    return boostrap_current


def find_j_max_from_boostrap_current(bootstrap_current: np.ndarray, slice_radius: np.ndarray): 
    # Create a boolean mask for slice_radius values within the desired range
    in_ped_mask = np.logical_and(slice_radius > 0.6, slice_radius < 1.1)

    # Use the mask to index bootstrap_current and slice_radius arrays
    bootstrap_current_in_ped = bootstrap_current[in_ped_mask]
    slice_radius_in_ped = slice_radius[in_ped_mask]
    if np.isnan(bootstrap_current_in_ped).sum() > 0: # TODO: THIS IS HACK SHOULD FIX SO THAT NO NANS ARE IN THE BOOSTRAP CURRENT
        bootstrap_current_in_ped = np.nan_to_num(bootstrap_current_in_ped, nan = -500000)
    # Find the index of the maximum value in bootstrap_current_in_ped
    max_idx = np.argmax(bootstrap_current_in_ped)
    
    # Use the max_idx to get the corresponding values of bootstrap_current and slice_radius
    max_bootstrap_current = bootstrap_current_in_ped[max_idx]
    max_slice_radius = slice_radius_in_ped[max_idx]
    # print(bootstrap_current_in_ped, max_bootstrap_current)
    # Find the index of max_slice_radius in the original slice_radius array
    max_pressure_grad_idx = np.where(slice_radius == max_slice_radius)[0][0]
    return max_bootstrap_current, max_slice_radius, max_pressure_grad_idx

