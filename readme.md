# Boosting the Measurement Capabilities of CANS by Statistical Inference

SANS is a commonly used technique for structural study of materials. However, its effectiveness is often perceived to depend on the availability of high incident neutron flux. In this work, we introduce a data analysis method grounded in the properties of multivariate Gaussian distributions, which accounts for the proximity relationships among detector pixels. This approach enhances the ability to identify trends obscured by statistical noise while simultaneously providing confidence intervals for measurements.
Our method demonstrates robustness against incoherent noise and accommodates variations in sample size and limited beam-time exposures. Crucially, it relies entirely on statistical inference from individual sparse measurements, eliminating the need for sample- or instrument-specific training datasets typically required for machine learning approaches. This innovation significantly enhances the measurement capabilities of CANS for structural studies using SANS.

![image](https://hackmd.io/_uploads/S1N2BRXFJx.png)

### Neutron Scattering with Limited Flux
Here we choose the $I(Q)$ of uniform hard sphere as example.
Theoretically, the probability distribution of neutrons arriving at the detector pixels is expected to be smooth, as illustrated in the figure.
> ![image](https://hackmd.io/_uploads/HkfpH2ZD1x.png)
> Probability distribution of neutrons arriving at each detector pixel

However, if the instrument flux is limited or the measurement time is very short, the data returned will be noisy. The following animation illustrates an example of varying the number of neutrons arriving at the detector from $10^2$ to $10^8$, following the distribution shown above. At least an average of $10^3$ neutrons per pixel is required to eliminate visible noise from the spectrum, which corresponds to a total of approximately $10^7$ neutrons per image.
> ![image](https://hackmd.io/_uploads/S1fC-4NOJx.png)
> ![2D_IQ](https://hackmd.io/_uploads/rkniv3Zw1x.gif)
> Sampled 2D spectrum at different total neutron counts


By performing a radial average, one can derive a typical 1D spectrum that might be observed under low flux conditions. Here, we observe that with only a total of $10^4$ neutrons per image:
* The data may have huge error bars.
* The data points do not form a smooth curve.
> ![image](https://hackmd.io/_uploads/Skb6SIgPyl.png)
> Radially averaged 1D spectrum.

This raises an important question: **How should the measured data be interpreted?**

The following figure presents independent samples (red curves) plotted over the measured intensity with error bars. Obviously, although the sampled $I(Q)$ for each $Q$ followed the distribution specified by the error bars, none of the red curves exhibit the ideal properties of **smoothness and continuity**, which we assume to underlie the probability distribution governing the measured data.
<!-- ![image](https://hackmd.io/_uploads/S1KUrjxPye.png) -->
> ![noisy_sample1d](https://hackmd.io/_uploads/Sya753Wvyl.gif)
> Independent samples from the distribution specified by the error bars.

<span style="color:red">"Smoothness and continuity"</span> imply the following statement to be valid: 
> **The scattering intensity at a Q-bin (or pixel) is highly correlated with its neighbors but loses correlation over larger separations.**

Our objective now is to construct a probability distribution that adheres to this statement while faithfully reflecting the information derived from experimental observations.
