Don't FRET
============


Don't FRET! is a python package featuring a web application for performing burst search on confocal smFRET data. 


## Process photon files

```
dont-fret process filename.ptu
```

Will process the file. It will perform burst search as specified in the configuration file. Output (default) are burst photons and bursts as .pq files. 

## Run the web application

To launch with the default configuration file:

```
dont-fret serve
```

To create a local default configuration file:

```
dont-fret config
```

Then you can edit the created config file. To launch the web application with a specific config file:

```
dont-fret serve --config config.yaml
```

## Configuration

Configuration for channels, photon streams and (default) burst search settings is done from the config .yaml file. 

First, define your channels:

```yaml
channels: # refactor channels in code to channel_identifiers
  laser_D:
    target: nanotimes
    value: [ 0, 1000 ]
  laser_A:
    target: nanotimes
    value: [ 1000, 2000 ] # intervals are inclusive, exclusive
  det_D:
    target: detectors
    value: 1
  det_A:
    target: detectors
    value: 0
```

Currently supported targets are `nanotimes`, `detectors` and `timestamps`. These are as read from the file and not converted to seconds. Modulo is supported (untested) for us-ALEX:
```yaml
channels: # refactor channels in code to channel_identifiers
  laser_D:
    target: timestamps
    value: [ 0, 100 ]
    modulo: 200
```

This will assign photons with a timestamp modulo 200 in the range from 0 up to 100 to "laser_D".

Next, define your photon streams. Photons streams are combinations of channels ("AND"):

```yaml
streams:
  DD: [laser_D, det_D]
  DA: [laser_D, det_A]
  AA: [laser_A, det_A]
  AD: [laser_A, det_D]
```

!IMPORTANT
The notation used here is excitation then emission, thus the FRET stream is 'DA' while in literature the FRET photon stream is often written as `A|D` (Acceptor emission during donor excitation). 


!IMPORTANT  
At the moment apparent FRET and stoichiometry are calculated from the defined photon streams and it is required the following streams are defined: 'AA', 'DD', 'DA' (=FRET). This is expected to be changed in future updates.

## Development

Download a test file:

```sh
wget https://kuleuven-my.sharepoint.com/:u:/g/personal/jochem_smit_kuleuven_be/Efy7ur779ARNiBlP05Ki7NMBabKX3auswj30xmpRLaIfPg?e=E6wWoZ&download=1
```


If autoreload (refresh web application upon code changes) doesnt work, run from:
solara run dont_fret\tmp.py -- --config default_testing.yaml

### Create a new release

- Create a new release on github. Create a new tag with the version (format: v0.1.0)
- github actions creates release on pypi
- done!