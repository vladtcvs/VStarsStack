# descriptor format

## angled descriptor

We have some star, star1 and star2

`(angle1, relative_size1, angle2, relative_size2, dangle)`

* angle1 - angle distance between star and star1
* relative_size1 - size1/size
* angle2 - angle distance between star and star2
* relative_size2 - size2/size
* dangle - angle between vector star->star1 and vector star->star2

## distance descriptor

We have some star and star1

`(angle1, size1/size, angle1, size1/size, 0)`

It can be useful if there only 2 bright objects on the image
