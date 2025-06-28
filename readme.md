# Installation

## From PyPa
```
python3 -m pip install vstarstack
```

## From sources
```
python3 setup.py install [--user]
```

# Modes of image presentation

* sphere mode
* flat mode

## Sphere mode

In this mode we use knowledge about celestial sphere and the fact, that out images are some projections of celestial sphere. So when we need to perform some movements of image for alignment, we have to use rotations of celestial sphere.

This mode should be used when perspective distorsions are valueable.

### Projections

* perspective projection - use perspective projection with specified focal length and pixel size
* equirectangular projection - used for planet surface map
* orhtographic projection - used for process planet surface images

## Flat mode

In this mode we don't know anything about celestial sphere and consider images as just flat images. We use standart movements of flat surface (rotation + shift) for alignment.


# Modes of image aligning

* stars mode - images contains stars and should be aligned by stars
* objects mode - images contains some small object, much less than image size, and this object
should be cutted out and images should be aligned to center object
* disc mode - used for Moon, and other big disc objects

## Stars mode

Command for work with stars beginning with `vstarstack stars`.

### detect

`vstarstack stars detect` - detect stars on image

### describe

`vstarstack stars describe`

We build descriptors for N most brightest stars. Each descriptor is invariant to rotations of image - it contains only information about distances to other stars, their relative brightness and angles between pairs of other stars. So we can use this descriptor for identifying stars.

### match

`vstarstack stars match`

We match the same stars on different images using descriptors abd build match table

## Objects

This commands work with non-star objects - planets, diffraction images, Moon, etc

`vstarstack objects config`
`vstarstack objects detect`
`vstarstack objects cut`
`vstarstack objects clusters`

## Planets

`vstarstack planets configure`
`vstarstack planets buildmap`

## Clusters

`vstarstack cluster`

# CI

Jenkins used for CI

# License

GNU GPLv3
