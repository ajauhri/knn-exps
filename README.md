Project Description
----
This is an implementation and a comparitive study of nearest-neighbour algorithms. NN algorithms are relevant for machine learning applications since it is believed that high-dimensoinal datasets have low-dimensional intrinsic structure. Specifically, we shall look at cover trees[1], and locality sensitive hashing[2] on different datasets and perform a comparitive study of their performance. Several datasets like user ratings, user tags on stackoverflow.com, and if possible, image classification will be considered. Moreover, from this study I aim to infer the correlation of a NN algorithm and structure of data via certain metrics of performance.


Dependencies
---
* [psutil](http://code.google.com/p/psutil/) 
* numpy v1.8.1
* [vlfeat](https://github.com/dougalsutherland/vlfeat-ctypes) 
* [guppy](https://pypi.python.org/pypi/guppy/0.1.9) for memory profiling 

Testing 
---
To run image patching use:

    ./img_patching data/a.png

The approximated image result will be saved as `res_img.eps` and the root mean squared error as `rms.txt`

To run LSH and cover trees with sparse data use:

    ./main.py -t data/netflix.mat -i 1000 -j 10 -n>o

To run LSH and cover trees with dense data use:

    ./main.py -t data/mnist.data -i 1000 -j 10 -g>o

References
----

[1] Alina Beygelzimer, Sham Kakade, and John Langford. Cover trees for nearest neighbor. In Proceedings of the
23rd international conference on Machine learning, pages 97–104. ACM, 2006.

[2] Mayur Datar, Nicole Immorlica, Piotr Indyk, and Vahab S Mirrokni. Locality-sensitive hashing scheme based
on p-stable distributions. In Proceedings of the twentieth annual symposium on Computational geometry, pages
253–262. ACM, 2004.

