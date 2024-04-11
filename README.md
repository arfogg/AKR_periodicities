# AKR_periodicities

This package is used to examine periodicities in Auroral Kilometric Radiation data.

**License:** CC0-1.0

**Support:** please [create an issue](https://github.com/arfogg/AKR_periodicities/issues) or contact [arfogg](https://github.com/arfogg) directly. Any input on the code / issues found are greatly appreciated and will help to improve the software.

## Required Packages

TBC

Developed using Python 3.8.8. 

## To Do:

General
- [ ] check Caitriona acknowledgement statement
- [ ] add in James acknowledgement statement
- [ ] create requirements.txt
- [ ] record required packages above

Code
- [x] add in docstrings

Data preparation
- [ ] deal with AKR data gaps / uneven temporal resolution

FFT work
- [x] get FFT to work on fake AKR intensity data
- [ ] automatically limit x axis of FFT
- [ ] deal with big peak at period=0
- [ ] repeat on the following subsets:
    * [ ] years, to investigate solar cycle dependence
    * [ ] Lamy 2010 and Waters 2021 / Cassini flyby interval
    * [ ] n random intervals of length = Cassini flyby interval or perhaps longer?
    * [ ] subsetting by spacecraft LT

## Acknowledgements

* ARF gratefully acknowledges the support of Irish Research Council Government of Ireland Postdoctoral Fellowship GOIPD/2022/782.
* CMJ gratefully acknowledges the support of Science Foundation Ireland Grant 18/FRL/6199.
