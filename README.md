# sloper 0.9.0
The main goal of this project (sloper.py) was to interpolate ASCII figure surface shape (normal vectors to the surface).
In addition it also contains simple physic engine (ascii_engine.py), that make use of surface data to simulate hail
(multi rigid bodies). All steps are precalculate before running animation. It use Euler method for integration
and "Speculative Contacts" for collision detection.

## ascii_engine.py (demos)
<p align="center">
<img src="./doc/demo1.gif"/>
<img src="./doc/demo3.gif"/>
</p>

## Requirements
```
$ pip install -r requirements.txt
```
Tested with:
- numpy            1.16.3
- tinyarray        1.2.1
- opencv-python    4.1.0.25
- Pillow           6.0.0


## Description
Sloper use very simple method to find surface shape (at least in this version), that's why any input data must be
pre-processed by user. First of all each figure must contain "marker" at the top-left corner in 3x3 area, that help sloper
calculate cell and grid size.
```
_^
^
```
Secondly, figure must be "drilled" from all characters that are not part of surface.

<p align="center">
<img src="./doc/umbrella_not_drilled.png"/>
<img src="./doc/umbrella_drilled.png"/>
</p>

When grid and markers are calculated, sloper find nearest neighbor to each character, and join them to get figure contour.

<p align="center">
<img src="./doc/ascii_image.png"/>
<img src="./doc/grid_and_markers.png"/>
<img src="./doc/contours.png"/>
</p>

Next for each point, where braille dot may appear, sloper calculate normal vector to the surface in this point.

<p align="center">
<img src="./doc/braille_dots.png"/>
<img src="./doc/normal_vectors.png"/>
</p>

## Usage
```
$ python sloper.py -a ascii_data/umbrella-drilled.txt
```

## References/Credits
- https://en.wikipedia.org/wiki/Collision_response#Impulse-based_reaction_model
- https://wildbunny.co.uk/blog/2011/03/25/speculative-contacts-an-continuous-collision-engine-approach-part-1/
- http://twvideo01.ubm-us.net/o1/vault/gdc2013/slides/824737Catto_Erin_PhysicsForGame.pdf
- https://github.com/mattleibow/jitterphysics/wiki/Speculative-Contacts
- https://github.com/pratik2709/Speculative-Contacts
- https://codepen.io/kenjiSpecial/pen/bNJQKQ
