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

After cloning the repository, download datasets using `wget`, 

    wget https://dl.dropboxusercontent.com/u/10119973/15-853/a.png 
    wget https://dl.dropboxusercontent.com/u/10119973/15-853/mnist.data 
    wget https://dl.dropboxusercontent.com/u/10119973/15-853/netflix.mat 

To run image patching use:

    ./img_patching.py data/a.png

The approximated image result will be saved as `res_img.eps` and the root mean squared error as `rms.txt` as `<rms_error>, <#of patches>`

To run LSH and cover trees with sparse data use:

    ./main.py -t data/netflix.mat -i 1000 -j 10 -n -r 30

To run LSH and cover trees with dense data use:

    ./main.py -t data/mnist.data -i 1000 -j 10 -g -r 1800

To get memory or timings results, please uncomment two functions `collect_timings` and `do_profiling` in `main.py`.

References
----

[1] Alina Beygelzimer, Sham Kakade, and John Langford. Cover trees for nearest neighbor. In Proceedings of the
23rd international conference on Machine learning, pages 97–104. ACM, 2006.

[2] Mayur Datar, Nicole Immorlica, Piotr Indyk, and Vahab S Mirrokni. Locality-sensitive hashing scheme based
on p-stable distributions. In Proceedings of the twentieth annual symposium on Computational geometry, pages
253–262. ACM, 2004.

