# descriptor format

## angled descriptor

We have some star, star1 and star2

`(id1, angle1, relative_size1, id2, angle2, relative_size2, dangle)`

* id1 - identifier of star1
* angle1 - angle distance between star and star1
* relative_size1 - size1/size
* id2 - identifier of star2
* angle2 - angle distance between star and star2
* relative_size2 - size2/size
* dangle - angle between vector star->star1 and vector star->star2

## distance descriptor

We have some star and star1

`(id1, angle1, size1/size, id1, angle1, size1/size, 0)`

It can be useful if there only 2 bright objects on the image
