# Cosmopower-in-Cosmosis

We developed cosmosis modules called comsopower_interface.py that uses a pretrained COSMOPOWER emulator found in the folder trained_models to predict the linear and non-linear power spectrum.
Since the COSMOPOWER is only used to predict the power spectra we additional need the camb_distance module, which is identical to the py_camb module without calcualting the power spectra. 

In order to to be able to use these modules you need to install:
pip install cosmopower

For a detailed description of COSMOPOWER see https://arxiv.org/pdf/2106.03846.pdf, where the pretrained models can also be found here https://github.com/alessiospuriomancini/cosmopower. 





