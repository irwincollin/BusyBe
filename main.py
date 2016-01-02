import urllib
import time

# My first python program (if you can call it that) of 2016! And my first github commit of 2016! Happy new year!
# In fact this is the start of my first serious python project of all time. Yay!

i = 1;
while i < 60:
    print "Downloading image number " + str(i);
    urllib.urlretrieve("http://recsports.ufl.edu/cam/cam2.jpg", str(i) + ".jpg")
    i = i + 1;
    time.sleep(30)
