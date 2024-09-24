# fip

This repository provides the implementation of fip, the **F**lood **I**nundation **P**arallel computation model.
> __[Accelerating flash flood simulations: An efficient GPU implementation for a slim shallow water solver](https://www.sciencedirect.com/science/article/pii/S1364815224000914)__  
> [Arne Rak](https://orcid.org/0000-0001-6385-3455), [Peter Mewis](https://orcid.org/0000-0002-4918-3202),  [Stefan Guthe](https://orcid.org/0000-0001-5539-9096)  
> _Environmental Modelling & Software, 2024_

> __[Massively Parallel Large Scale Inundation Modelling](https://diglib.eg.org/items/d6320856-9753-49c4-a970-76722f6ca1f1)__  
> [Arne Rak](https://orcid.org/0000-0001-6385-3455),  [Stefan Guthe](https://orcid.org/0000-0001-5539-9096), [Peter Mewis](https://orcid.org/0000-0002-4918-3202)  
> _EGPGV@ EuroVis, 2022_  

## Windows compilation
You can build FIP on Windows using CMake and Visual Studio Community Edition. CUDA toolkit has to be installed in version 11.4 or above.

Precompiled binaries for Windows can be found in the [Releases section](https://github.com/fip-ems/fip/releases).
## Linux compilation steps
FIP may work fine with CUDA toolkit versions above 11.4. The following is an installation example for CUDA toolkit version 11.4 on Ubuntu 18.04. 
### CUDA Toolkit 11.4 installation
Disable nouveau kernel driver
```
sudo echo -e "blacklist nouveau\noptions nouveau modeset=0" >> /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u
sudo reboot
```

Install CUDA toolkit 11.4

```
wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda_11.4.0_470.42.01_linux.run
sudo sh cuda_11.4.0_470.42.01_linux.run
# Add to .bashrc
echo 'PATH="/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
echo 'LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
```

### Project compilation

```
sudo apt install cmake build-essential
mkdir bin; cd bin
cmake ..
make -j
```
## Running FIP
The simulation parameters are given in a YAML file as the first launch argument of FIP `./fip parameters.yml`. The following table lists all simulation parameters and their usage:
|Parameter| Type | Required | Example | Description |
|--|--| --| -- | -- |
| `name`        | string | yes | `name: my_simulation` | Name of the simulation. Used when writing out simulation buffers.
| `terrain`     | [string, offset] | yes | `terrain: [marchfeld_sohle, 36]` | Binary DTM file. `offset` is the byte offset at which the data array starts.
| `W`           | integer| yes | `W: 5667` | Number of grid cells in x direction.
| `H`           | integer| yes | `H: 6500` | Number of grid cells in y direction.
| `dx`          | float | yes | `dx: 3` | Grid cell resolution in meters.
| `dt`          | float | yes | `dt: 0.1` | Initial timestep in seconds.
| `duration`    | float | yes | `duration: 3600*24*12` | Simulation duration in seconds. Only used when run without GUI. Can be a math expression using only multiplications.
| `kSt` *or*<br> `kSt_var`| float *or*<br> [string, offset] | yes | `kSt: 30.0` *or*<br>`kSt_var: [marchfeld_kst, 36]` | Strickler roughness coefficient.
| `z`           | [string, offset] | optional | `z: [marchfeld_init_z, 36]` | Initial water levels.
| `qx`          | [string, offset] | optional | `qx: [marchfeld_init_qx, 36]` | Initial discharges in x direction.
| `qy`          | [string, offset] | optional | `qy: [marchfeld_init_qy, 36]` | Initial discharges in y direction.
| `variable_dt` | boolean | optional | `variable_dt: true` | When `true`, the timestep dt will adjust during simulation according to the CFL condition.
| `no_data`| float | optional | `no_data: -1` | When set, cells in the DTM with this value will be ignored in rendering and during computation.
| `save_state`| float[] | optional | <pre>save_state:<br>&nbsp; - 3600 * 24 * 0.5<br>&nbsp; - 3600 * 24 * 2.5</pre> | The GPU buffers for z, qx, qy and flood plains will be written to disk at the given timestamps.
| `sampling`| { label: string, <br> type: z \| qx \| qy, <br> x: integer, <br> y: integer }[] | optional |  <pre>sampling:<br>&nbsp; - x: 0<br>&nbsp;&nbsp;&nbsp; y: 170<br>&nbsp;&nbsp;&nbsp; label: sample1<br>&nbsp;&nbsp;&nbsp; type: z<br>&nbsp;&nbsp;- { x: 42, y: 170, <br>&nbsp;&nbsp;&nbsp;&nbsp; label: sample2, type: qx } </pre> | At the given `sample_interval`, all sampling points in the `sampling` list will be printed to the standard output in CSV format. `x,y` define the cell coordinate, `type` defines which buffer is sampled.
| `sampling_interval`| integer | optional | `sampling_interval: 100` | Sampling interval in seconds.
| `boundary_conditions`| { side: right \| left \| top \| bottom, <br> type: z \| q \| close \| open, <br>z: float \| csv, <br>q: float \| csv, <br> from: integer, <br> to: integer }[] | optional | <pre>boundary_conditions:<br>&nbsp; - side: left<br>&nbsp;&nbsp;&nbsp; type: z<br>&nbsp;&nbsp;&nbsp; z: [timeseries.csv, 0, 1]<br>&nbsp; - side: right<br>&nbsp;&nbsp;&nbsp; type: close<br>&nbsp;&nbsp;&nbsp; from: 0<br>&nbsp;&nbsp;&nbsp; to: 500</pre> | Boundary conditions that drive the simulation when no initial water levels are given. `side` defines at which grid border the condition is applied. `from, to` are optional and define the range of cells at which the condition is applied. When omitted, condition is applied to entire border. `type` can be `open` (water flows out), `close` (walled off), a water level `z`, or discharge `q`. `z,q` can be fixed values or timeseries in CSV format given as `[filename, time-column, data-column]`. 


