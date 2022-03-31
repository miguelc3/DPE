This folder contains: 
- IMAGES: The original images that contains 5 sub-folders with the images obtained with the 5 different sensors. 
    The names of the images follow this pattern: Im_[GT|noGT]_<Name of the sensor>_<number of the image>.<format>. 
    - GT/noGT means with Ground Truth/without Ground Truth. 
    - The name of the 5 sensors are: AIGLE_RN, ESAR, LCMS, LRIS, TEMPEST2. 
- GROUNDTRUTH: The ground truth of some of the images of IMAGES (whose named ***_GT_***) per sensors (so, again, 5 sub-folders). 
    The names of the ground truth follow this pattern: GT_<Name of the sensor>_<number of the image>.<format>. 
    The names of the 5 sensors are still the same. 
- RESULTS: It contains all the results obtained for each image (with and wihtout ground truth), for all the approaches tested. 
    So, it also contains 5 sub-folders. 
    The names of the results follow this pattern: Res_[GT/noGT]_<Name of the sensor>_<number of the image>_<name of the method>.<format>
    GT/noGT means the same as for original images. 
    Name of the sensor are still the same. 
    Name of the method are :
    - GC for Geodesic Contours based on Points Of Interest detection [Chambon2011a]
    - M1 and M2 for the Markovian Modelling and Adaptive Filtering approaches [Chambon2011b]
    - MPS for the Minimal Path Selection method [Amhaz 2016] 
    - NGU for Free Form Anisotropy [Nguyen 2011]
- AmhazChambon_TITS_2016.pdf: the draft version of the paper related to this dataset. 

If you have any questions, feel free to contact me: schambon@enseeiht.fr

------------------------------------------------------------------------

Bibliography : 

[Chambon2011a] Sylvie Chambon. 
" Detection of Points of Interest for Geodesic Contours: Application on Road Images for Crack Detection ". 
In International Joint Conference on Computer Vision Theory and Applications, VISAPP, 
2011, Vilamoura, Portugal. 

[Chambon2011b] Sylvie Chambon, Jean-Marc Moliard. 
" Automatic Road Pavement Assessment with Image Processing: Review and Comparison ". 
International Journal of Geophysics, 
2011, volume 2011, Article ID 989354, 20 pages. 

[Amhaz 2016] R. Amhaz, S. Chambon, J. Idier, V. Baltazart. 
Automatic crack detection on 2D pavement images: An algorithm based on minimal path selection. 
IEEE Transactions on Intelligent Transport Systems. 
2016. To appear. 

[Nguyen 2011] T. S. Nguyen, S. Begot, F. Duculty, and M. Avila. 
"Free-form anisotropy: A new method for crack detection on pavement surface images",
In International Conference on Image Processing, 
2011, pp. 1069–1072. 
