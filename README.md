## Adaptive Coding in Wireless Acoustic Sensor Networks for Distributed Blind System Identification

[M. Blochberger](https://orcid.org/0000-0001-7331-7162), [J. Østergaard](https://orcid.org/0000-0002-3724-6114), [F. Elvander](https://orcid.org/0000-0003-1857-2173), [R. Ali](https://orcid.org/0000-0001-7826-1030), J. Jensen, [M. Moonen](https://orcid.org/0000-0003-4461-0073), [T. van Waterschoot](https://orcid.org/0000-0002-6323-7350)

## Abstract
With distributed signal processing gaining traction in the audio and speech processing landscape through the utilization of interconnected devices constituting wireless acoustic sensor networks, additional challenges arise, including optimal data transmission between devices (nodes). In this contribution, we extend an adaptive distributed blind system identification algorithm by introducing a residual-based adaptive coding scheme to minimize communication costs within the network. We introduce a coding scheme that utilizes residuals to take advantage of the convergence of estimates, i.e., vanishing residuals, to minimize information being sent. Further, we adapt the quantization of the residuals to their variance during the algorithm’s steady state, for which we use time-recursive variance estimates. We demonstrate the approach’s effectiveness with numerical simulations and its performance in various scenarios.

## Archived paper
TODO

## Running instructions

- [Install Docker](https://www.docker.com/)
- Clone this repository: `git clone https://github.com/mbl-sounds/asilomar2023sim`
- Build the docker image: `docker build -t asilomar2023sim/simulations .`
- Run the docker image: `docker run -it --rm -v "$(pwd)/.:/wd" asilomar2023sim/simulations`
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
