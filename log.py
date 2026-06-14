
""" Timers and log for SWE-solver space + time operators 
"""
#-- Part of the PERISCOPE solver
#-- Darren Engwirda
#-- d.engwirda@gmail.com
#-- https://github.com/dengwirda/

class base: pass
tcpu = base()
tcpu.evaluate_ = 0.0E+00  # integration
tcpu.thickness = 0.0E+00
tcpu.momentum_ = 0.0E+00
tcpu.finalise_ = 0.0E+00
tcpu.filewrite = 0.0E+00
tcpu.calc_obcs = 0.0E+00  # rhs
tcpu.calc_udry = 0.0E+00
tcpu.upwinding = 0.0E+00
tcpu.calc_hmap = 0.0E+00
tcpu.calc_qmap = 0.0E+00
tcpu.tend_hadv = 0.0E+00
tcpu.tend_qadv = 0.0E+00
tcpu.calc_u_ke = 0.0E+00
tcpu.calc_u_pv = 0.0E+00
tcpu.tend_uadv = 0.0E+00
tcpu.tend_upgf = 0.0E+00
tcpu.calc_perp = 0.0E+00
tcpu.calc_umix = 0.0E+00  # dissipation
tcpu.calc_uwav = 0.0E+00
tcpu.calc_hmix = 0.0E+00
tcpu.tend_umix = 0.0E+00
tcpu.tend_hmix = 0.0E+00
tcpu.calc_tide = 0.0E+00  # extern. frc
tcpu.calc_self = 0.0E+00
tcpu.tend_ugeo = 0.0E+00
tcpu.tend_utau = 0.0E+00
tcpu.calc_drag = 0.0E+00

