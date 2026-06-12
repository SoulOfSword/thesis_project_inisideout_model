"""JAX galaxy model engine for both accretion scenarios."""

from .common import (M_times1, M_times1_jax, N_T, T_END, DT,
                     log_M_bar_array, log_M_bar_array_jax,
                     C_def_jax, simpson_uniform_jax, interp1d_jax,
                     full_from_sigma_jax, full_from_sigma_jax_mcmc)
from .inside_out import (build_r_acc_matrix_for_all_M, build_r_acc_matrix_for_all_M_jax,
                         Sigma_definer_jax, Full_final_definer_jax, Full_final_definer_jax_mcmc,
                         run_all_masses, all_obs_for_galaxies_jax, fgas_and_jbar_for_galaxies_jax,
                         F_omega_jax, solve_omega_bisect_autobracket_jax)
from .non_inside_out import (omega_Mdep, tacc_Mdep, Sigma_definer_static_racc_jax,
                             Full_final_definer_Mdep_omega_jax, run_all_masses_Mdep_omega_jax,
                             build_r_acc_for_single_M, run_single_galaxy_from_jbar)
from .profiles import (radial_profiles_io, radial_profiles_nio, radial_profiles_nio_core,
                       save_radial_profiles, load_radial_profiles)

__all__ = [
    "M_times1", "M_times1_jax", "N_T", "T_END", "DT",
    "log_M_bar_array", "log_M_bar_array_jax",
    "C_def_jax", "simpson_uniform_jax", "interp1d_jax",
    "full_from_sigma_jax", "full_from_sigma_jax_mcmc",
    "build_r_acc_matrix_for_all_M", "build_r_acc_matrix_for_all_M_jax",
    "Sigma_definer_jax", "Full_final_definer_jax", "Full_final_definer_jax_mcmc",
    "run_all_masses", "all_obs_for_galaxies_jax", "fgas_and_jbar_for_galaxies_jax",
    "F_omega_jax", "solve_omega_bisect_autobracket_jax",
    "omega_Mdep", "tacc_Mdep", "Sigma_definer_static_racc_jax",
    "Full_final_definer_Mdep_omega_jax", "run_all_masses_Mdep_omega_jax",
    "build_r_acc_for_single_M", "run_single_galaxy_from_jbar",
    "radial_profiles_io", "radial_profiles_nio", "radial_profiles_nio_core",
    "save_radial_profiles", "load_radial_profiles",
]
