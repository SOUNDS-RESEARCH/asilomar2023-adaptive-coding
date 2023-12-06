## Adaptive Coding in Wireless Acoustic Sensor Networks for Distributed Blind System Identification

[M. Blochberger¹](https://orcid.org/0000-0001-7331-7162), [J.
Østergaard²](https://orcid.org/0000-0002-3724-6114), [R.
Ali³](https://orcid.org/0000-0001-7826-1030), [M.
Moonen¹](https://orcid.org/0000-0003-4461-0073), [F.
Elvander⁴](https://orcid.org/0000-0003-1857-2173), J. Jensen², [T. van
Waterschoot¹](https://orcid.org/0000-0002-6323-7350)
    
¹ KU Leuven  
² Aalborg University  
³ University of Surrey  
⁴ Aalto University  

## Abstract
With distributed signal processing gaining traction in the audio and speech processing
landscape through the utilization of interconnected devices constituting wireless acoustic
sensor networks, additional challenges arise, including optimal data transmission between
devices. In this paper, we extend an adaptive distributed blind system identification
algorithm by introducing a residual-based adaptive coding scheme to minimize communication
costs within the network.  We introduce a coding scheme that takes advantage of the
convergence of estimates, i.e., vanishing residuals, to minimize information being sent.
The scheme is adaptive, i.e., tracks changes in the estimated system and utilizes entropy
coding and adaptive gain to fit the time-varying residual variance to pre-trained
codebooks. We use a low-complexity approach for gain adaptation, based on a recursive
variance estimate. We demonstrate the approach's effectiveness with numerical simulations
and its performance in various scenarios.  

## Running instructions

- [Install Docker](https://www.docker.com/)
- Clone this repository: `git clone https://github.com/SOUNDS-RESEARCH/asilomar2023-adaptive-coding --recurse-submodules`
- Build the docker image: `docker build -t asilomar2023sim/simulations .`
- Run the docker image: `docker run -it --rm -v "$(pwd)/.:/wd" asilomar2023sim/simulations <nr_mc_runs> <random_seed> <number_of_processes>`
- The results can be found in the `data/` directory

## SOUNDS
This research work was carried out at the ESAT Laboratory of KU Leuven and the Section of AI and Sound of Aalborg University as part of the SOUNDS European Training Network.

[SOUNDS Website](https://www.sounds-etn.eu/)

## Acknowledgements
<table>
    <tr>
        <td width="75">
        <img src="https://www.sounds-etn.eu/wp-content/uploads/2021/01/Screenshot-2021-01-07-at-16.50.22-600x400.png"  align="left"/>
        </td>
        <td>
        This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 956369
        </td>
    </tr>
</table>
