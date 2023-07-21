# __BBct__

**[7th Super Computing Youth Camp 2023](https://sites.google.com/view/scyouthcamp/) Results**

***
### LICENSE
**BSD License: Include in source code or documentation and other materials the disclaimer of copyright presentation, conditions of compliance, and disclaimer of warranty when redistributing
Failure to use the name of the first developer or contributor in warranty or promotion of the product.**

***
### Abstract
* Hosted by: UNIST, KISTI

* Duration: 2023-07-17 to 2023-07-21 (5 days)

* Team name: BBct

* Members: Chan Woo Park, Tae Kyu Baek, Ji Ho Baek

* Contents: 
 * Ray Tracing Parallel Code
 * Gender-Differentiated Artificial Intelligence (TensorFlow)

***

# ✔️ Ray Tracing Parallel Code

### How to USE

1. Download [Anaconda](https://www.anaconda.com/)
2. Run Anaconda Prompt
3. Install MPI `` conda create -n mpi mpi4py numpy Pillow``
5. Activate Conda `` conda activate mpi``
6. Set up a directory ``cd C:\..``
7. Run file ``mpiexec -n [number of process] python ray_mpi.py
8. Find image3.png

***

### Code Description

1. Library and Function Definitions: Import the ``mpi4py``, ``numpy``, and ``matplotlib.pyplot`` libraries, and define functions required for ray tracing, such as ``normalize``, ``reflected``, ``sphere_intersect``, ``nearest_intersected_object``, and ``ray_tracing``.

2. MPI Initialization: Initialize the MPI environment using ``comm = MPI.COMM_WORLD``. Obtain the number of available processes with ``size = comm.Get_size()``, and get the rank of the current process with ``rank = comm.Get_rank()``.

3. Definition of Ray Tracing Parameters and Objects: Define parameters required for ray tracing (width, height, camera position, light position, etc.) and objects (spheres) in the ``objects`` list.

4. Image Partitioning: Divide the image by the number of available processes (``height``) to distribute the ray tracing workload among different processes. Each process will perform ray tracing only for its assigned pixels.

5. Parallel Ray Tracing: Each process performs ray tracing for its assigned region of the image, and the results are stored in the ``image`` array.

6. Result Collection and Image Saving: Use the ``comm.Gatherv()`` function to gather results from each process into ``recvbuf``, and then use ``plt.imsave()`` to save the final image.

7. Measurement of Execution Time: Measure the overall execution time of the ray tracing and print it for analysis.
