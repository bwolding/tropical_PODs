# plume_model
A simple model to compute plume thermodynamic properties upon lifting and mixing

Ensure that python is installed on your machine with requisite libraries.

Steps to run plume model:
  i) Ensure that you have access to ARM_Nauru directory (for input files). 
       Else modify code to point to right input
  ii)  Compile the cython libraries with "python setup.py --build_ext -inplace" in your terminal ### Brandon's note: I had to run "python setup.py build_ext --inplace". First make sure to run "source /glade/u/home/bwolding/ncar_pylib_clone_20190723_Casper/bin/activate"
  iii) If cython build is successful, run "python plume_model.py" in your terminal. 
  
  
  
  BRANDON'S NOTES 06/22/2022
  
  In order to Cythonize:
  1.) Load Python clone using "source /glade/u/home/bwolding/ncar_pylib_clone_20190723_Casper/bin/activate"
  2.) Load GNU compiler using "module load gnu/9.1.0"
  3.) Setup using "python setup.py build_ext --inplace" which should give an ".so" file

