# Modes of image presentation

* sphere mode
* flat mode

## Sphere mode

In this mode we use knowledge about celestial sphere and the fact, that out images are some projections of celestial sphere. So when we need to perform some movements of image for alignment, we have to use rotations of celestial sphere.

This mode should be used when perspective distorsions are valueable.

### Projections

* perspective projection - use perspective projection with specified focal length and pixel size

## Flat mode

In this mode we don't know anything about celestial sphere and consider images as just flat images. We use standart movements of flat surface (rotation + shift) for alignment.


# Modes of image aligning

* stars mode - images contains stars and should be aligned by stars
* compact_objects - images contains some small object, much less than image size, and this object
should be cutted out and images should be aligned to center object

## Stars mode

Command for work with stars beginning with `starstack stars`.

### detect

`starstack stars detect` - detect stars on image

### lonlat

`starstack stars lonlat`

If we use `sphere` mode, we transform `(y,x)` coordinates of detected stars into `(lat,lon)` coordinates, with `(0,0)` at center of image.

### describe

`starstack stars describe`

We build descriptors for N most brightest stars. Each descriptor is invariant to rotations of image - it contains only information about distances to other stars, their relative brightness and angles between pairs of other stars. So we can use this descriptor for identifying stars.

### match

`starstack stars match`

We match the same stars on different images using descriptors

### net

`starstack stars net`

Build `net.json` - file with info about star matching. This is intermidiate format.

### cluster

`starstack stars cluster`

Build `clusters.json` - file with clusters of stars. It contains info about stars coordinates on each frame.

### process

`starstack stars process` - do all steps above in a single run

# License

GNU GPLv3
